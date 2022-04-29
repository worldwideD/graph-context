import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss

num_layers = 2
n_heads = 8

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear((num_layers + 1) * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear((num_layers + 1) * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        assert config.hidden_size % n_heads == 0

        self.GAT = MultilayersGAT(in_feat=config.hidden_size, nlayers=num_layers, out_feat=config.hidden_size, e_feat=config.hidden_size, nhid=config.hidden_size // n_heads, dropout=0.0, alpha=0.2, nheads=n_heads)
    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention
    
    def get_graph(self, sequence_output, attention, entity_pos):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        m_cnt, e_cnt, d_cnt, m_node, m_sum = 0, 0, len(entity_pos), [], []
        n, h, _, c = attention.size()
        # get mention info  
        for i in range(d_cnt):
            sum = 0
            for e in entity_pos[i]:
                for start, end, sent_id in e:
                    if start + offset < c:
                        # In case the entity mention is truncated due to limited max seq length.
                        m_emb = sequence_output[i, start + offset]
                        m_att = attention[i, :, start + offset]
                        m_node.append({'emb': m_emb, 'att': m_att, 'id': m_cnt, 'sent': sent_id, 'entity': e_cnt, 'doc': i})
                        m_cnt += 1
                        sum += 1
                e_cnt += 1
            m_sum.append(sum)
        
        #  build adj matrix and edge id
        adj, node_features, node_att, edge_id = [], [], [], []
        for m in m_node:
            node_features.append(m["emb"])
            node_att.append(m["att"])
            _adj = []

            for _m in m_node:
                if m["doc"] == _m["doc"] and m["sent"] == _m["sent"]: # add a self-loop here
                    _adj.append(2)
                else:
                    _adj.append(int(m["entity"] == _m["entity"]))
                edge_id.append([m["id"], _m["id"]])
            
            for j in range(d_cnt):
                _adj.append(int(m["doc"] == j))
                edge_id.append([m["id"], m_cnt+j])

            adj.append(_adj)
        
        for i in range(d_cnt):
            _adj = []
            
            for m in m_node:
                _adj.append(int(m["doc"] == i))
                edge_id.append([m_cnt+i, m["id"]])
            for j in range(d_cnt):
                _adj.append(int(i == j))
                edge_id.append([m_cnt+i, m_cnt+j])
            
            adj.append(_adj)
            node_features.append(sequence_output[i, 0])
            att = torch.ones(h, c).to(attention)
            node_att.append(att)
        
        # calculate attention
        node_att = torch.stack(node_att, dim=0)  # [node_num, h, c]
        edge_id = torch.LongTensor(edge_id).to(sequence_output.device)  # [node_num*node_num, 2]
        h_att = torch.index_select(node_att, 0, edge_id[:, 0])  # [node_num*node_num, h, c]
        t_att = torch.index_select(node_att, 0, edge_id[:, 1])
        ht_att = (h_att * t_att).mean(1)
        ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)  # [node_num*node_num, c]

        n_node = m_cnt+d_cnt
        e_features = None
        ht_att = ht_att.view(n_node, n_node, c)
        off = 0
        for i in range(d_cnt):
            le = torch.zeros(m_sum[i], off, self.config.hidden_size).to(attention)

            att = ht_att[off:off+m_sum[i], off:off+m_sum[i]].reshape(m_sum[i] * m_sum[i], -1)
            ctx = contract("ld,rl->rd", sequence_output[i], att)
            mide = ctx.view(m_sum[i], m_sum[i], self.config.hidden_size)

            re = torch.zeros(m_sum[i], n_node-(off+m_sum[i]), self.config.hidden_size).to(attention)
            _e = torch.cat([le, mide, re], dim=1)
            if e_features == None:
                e_features = _e
            else:
                e_features = torch.cat([e_features, _e], dim=0)
            off += m_sum[i]
        _e = torch.zeros(d_cnt, n_node, self.config.hidden_size).to(attention)
        if e_features == None:
            e_features = _e
        else:
            e_features = torch.cat([e_features, _e], dim=0)
        # return
        adj_ = torch.tensor(adj).to(sequence_output.device)
        node_features_ = torch.stack(node_features, dim=0)
        return adj_, node_features_, e_features

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end, sent_id in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end, sent_id = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss
    
    def get_new_ht(self, sequence_output, attention, g_features, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss = [], []
        for i in range(len(entity_pos)):
            entity_embs = []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb = []
                    for start, end, sent_id in e:
                        if start + offset < c:
                            e_emb.append(g_features[0])
                            g_features = g_features[1:]
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                else:
                    start, end, sent_id = e[0]
                    if start + offset < c:
                        e_emb = g_features[0]
                        g_features = g_features[1:]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                entity_embs.append(e_emb)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]    

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            
            hss.append(hs)
            tss.append(ts)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        return hss, tss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        adj, g_features, e_features = self.get_graph(sequence_output, attention, entity_pos)
        g_features = self.GAT(g_features, adj, e_features)

        #hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
        hs, ts = self.get_new_ht(sequence_output, attention, g_features, entity_pos, hts)

        #hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        #ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        hs = torch.tanh(self.head_extractor(hs))
        ts = torch.tanh(self.tail_extractor(ts))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output

class MultilayersGAT(nn.Module):
    def __init__(self, in_feat, out_feat, e_feat, nhid, alpha, nheads, dropout=0.0, nlayers=2):
        super().__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        
        self.GATlayers = nn.ModuleList([GAT(in_feat, out_feat, nhid, e_feat, dropout=dropout, alpha=alpha, nheads=nheads) for _ in range(nlayers)])

    def forward(self, x, adj, e):
        h = [x]
        for GATlayer in self.GATlayers:
            x = GATlayer(x, adj, e)
            h.append(x)
        h = torch.cat(h, dim=1)

        return h

class GAT(nn.Module):
    def __init__(self, in_features, out_features, hid_features, e_features, alpha, nheads, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.out_features = out_features

        self.attns = nn.ModuleList([GraphAttentionLayer(in_features, hid_features, e_features, dropout=dropout, alpha=alpha) for _ in range(nheads)])
        #self.W = nn.Parameter(torch.zeros(size=(nheads * hid_features, out_features, )))
        #nn.init.xavier_uniform_(self.W.data, gain=1.414)
    
    def forward(self, x, adj, e):
        #x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, e) for att in self.attns], dim=1)
        #x = F.dropout(x, self.dropout, training=self.training)
        
        N = x.size()[0]
        #y = x.view(N, self.nheads, self.out_features).mean(1)
        #y = F.elu(torch.matmul(x, self.W))
        x = F.elu(x)

        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, e_features, alpha, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.e_features = e_features
        self.alpha = alpha

        self.W_q = nn.Parameter(torch.zeros(size=(in_features, out_features, )))
        nn.init.xavier_uniform_(self.W_q.data, gain=1.414)
        self.W_k = nn.Parameter(torch.zeros(size=(in_features + e_features, out_features, )))
        nn.init.xavier_uniform_(self.W_k.data, gain=1.414)
        self.W_v = nn.Parameter(torch.zeros(size=(in_features + e_features, out_features, )))
        nn.init.xavier_uniform_(self.W_v.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, input, adj, e_feat):
        hq = torch.mm(input, self.W_q)
        N = hq.size()[0]
        hk = torch.cat([input.repeat(N, 1), e_feat.view(N * N, -1)], dim=1)
        hv = torch.matmul(hk, self.W_v)
        hk = torch.matmul(hk, self.W_k)

        a_input = torch.cat([hq.repeat(1, N).view(N * N, -1), hk], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        
        att = torch.where(adj > 0, e, zero_vec)
        att = F.softmax(att, dim=1).unsqueeze(1)
        hv = hv.view(N, N, -1)
        h_prime = torch.matmul(att, hv).squeeze(1)

        return h_prime