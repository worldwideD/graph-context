import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.GAT = MultilayersGAT(in_feat=config.hidden_size, nlayers=2, out_feat=config.hidden_size, nhid=128, dropout=0.0, alpha=0.2, nheads=8)

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
                if len(e) > 1:
                    for start, end, sent_id in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            m_emb = sequence_output[i, start + offset]
                            m_node.append({'emb': m_emb, 'id': m_cnt, 'sent': sent_id, 'entity': e_cnt, 'doc': i})
                            m_cnt += 1
                            sum += 1
                else:
                    start, end, sent_id = e[0]
                    if start + offset < c:
                        m_emb = sequence_output[i, start + offset]
                        m_node.append({'emb': m_emb, 'id': m_cnt, 'sent': sent_id, 'entity': e_cnt, 'doc': i})
                        m_cnt += 1
                        sum += 1
                e_cnt += 1
            m_sum.append(sum)
        
        #  build adj matrix and edge id
        adj, node_features= [], []
        for m in m_node:
            node_features.append(m["emb"])
            _adj = []

            for _m in m_node:
                _adj.append(int(m["entity"] == _m["entity"] or (m["doc"] == _m["doc"] and m["sent"] == _m["sent"])))
            
            for j in range(d_cnt):
                _adj.append(int(m["doc"] == j))

            adj.append(_adj)
        
        for i in range(d_cnt):
            feature, _adj = [], []
            
            for m in m_node:
                _adj.append(int(m["doc"] == i))
            for j in range(d_cnt):
                _adj.append(int(i == j))
            for j in range(c):
                feature.append(sequence_output[i, j])
            
            adj.append(_adj)
            node_features.append(torch.logsumexp(torch.stack(feature, dim=0), dim=0))

        n_node = m_cnt+d_cnt
        e_features = []
        
        # return
        adj_ = torch.tensor(adj).to(sequence_output.device)
        node_features_ = torch.stack(node_features, dim=0)
        return adj_, node_features_

    def get_ht(self, sequence_output, attention, g_features, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, m_cnt = [], [], 0
        for i in range(len(entity_pos)):
            entity_embs = []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb = []
                    for start, end, sent_id in e:
                        if start + offset < c:
                            e_emb.append(g_features[m_cnt])
                            m_cnt += 1
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                else:
                    start, end, sent_id = e[0]
                    if start + offset < c:
                        e_emb = g_features[m_cnt]
                        m_cnt += 1
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
        adj, g_features = self.get_graph(sequence_output, attention, entity_pos)
        g_features = self.GAT(g_features, adj)

        #hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
        hs, ts = self.get_ht(sequence_output, attention, g_features, entity_pos, hts)

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
    def __init__(self, in_feat, out_feat, nhid, alpha, nheads, dropout=0.0, nlayers=3):
        super().__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        
        self.GATlayers = nn.ModuleList([GAT(in_feat if _ == 0 else nhid * nheads, out_feat if _ == nlayers-1 else nhid, dropout=dropout, alpha=alpha, nheads=nheads, islast=(_ == nlayers-1)) for _ in range(nlayers)])

    def forward(self, x, adj):
        for GATlayer in self.GATlayers:
            x = GATlayer(x, adj)
        return x

class GAT(nn.Module):
    def __init__(self, in_features, out_features, alpha, nheads, dropout=0.0, islast=False):
        super().__init__()
        self.dropout = dropout
        self.islast = islast
        self.nheads = nheads
        self.out_features = out_features

        self.attns = nn.ModuleList([GraphAttentionLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=not islast) for _ in range(nheads)])
        #self.attns = [GraphAttentionLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=not islast) for _ in range(nheads)]
        #for i, attn in enumerate(self.attns):
        #    self.add_module('attention_{}'.format(i), attn)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attns], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        if (self.islast == True):
            N = x.size()[0]
            x = x.view(N, self.nheads, self.out_features).mean(1)
            x = F.elu(x)
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha, dropout=0.0, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features, )))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        
        att = torch.where(adj > 0, e, zero_vec)
        att = F.softmax(att, dim=1)
        att = F.dropout(att, self.dropout, training=self.training)
        h_prime = torch.matmul(att, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
