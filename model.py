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

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)

        self.rel = nn.Parameter(torch.zeros(size=(config.num_labels, 2 * emb_size, )))
        nn.init.xavier_uniform_(self.rel.data)
        self.bias = nn.Parameter(torch.zeros(size=(config.num_labels, )))
        nn.init.zeros_(self.bias)
        self.W = nn.Parameter(torch.zeros(size=(2 * emb_size, 2 * emb_size)))
        nn.init.xavier_uniform_(self.W.data)

        #self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

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

    def get_hrt(self, sequence_output, attention, entity_pos, htms, cut):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss, cuts = [], [], [], []
        mx = max(cut)
        ## get mention pairs emb
        for i in range(len(entity_pos)):
            mention_embs, mention_atts = [], []
            m_cnt = 0
            for e in entity_pos[i]:
                st = m_cnt
                for start, end in e:
                    if start + offset < c:
                        mention_embs.append(sequence_output[i, start + offset])
                        mention_atts.append(attention[i, :, start + offset])
                        m_cnt += 1
                    else:
                        mention_embs.append(torch.zeros(self.config.hidden_szie).to(sequence_output))
                        mention_atts.append(torch.zeros(h, c).to(attention))
                        m_cnt += 1
                if st == m_cnt:
                    mention_embs.append(torch.zeros(self.config.hidden_szie).to(sequence_output))
                    mention_atts.append(torch.zeros(h, c).to(attention))
                    m_cnt += 1

            mention_embs = torch.stack(mention_embs, dim=0)
            mention_atts = torch.stack(mention_atts, dim=0)

            ht_i = torch.LongTensor(htms[i]).to(sequence_output.device)
            hs = torch.index_select(mention_embs, 0, ht_i[:, 0])
            ts = torch.index_select(mention_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(mention_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(mention_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
            cuts = cuts + cut[i]
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss, cuts


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                htms=None,
                cut=None,
                mplabels=None,
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts, cuts = self.get_hrt(sequence_output, attention, entity_pos, htms, cut)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        '''
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        '''

        m_rep = torch.cat([hs, ts], dim=1)
        m_rep = torch.split(m_rep, cuts, dim=0)
        q = torch.matmul(self.rel, self.W)
        def get_e_rep(m):
            attn = torch.matmul(m, q.unsqueeze(2)).squeeze(2)
            attn = F.softmax(attn, dim=-1)
            e_rep = contract("ld,rl->rd", m.float(), attn)
            return e_rep
        e_rep = torch.stack(list(map(get_e_rep, m_rep))).to(sequence_output)
        logits = torch.mul(e_rep, self.rel).sum(2) + self.bias

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output
