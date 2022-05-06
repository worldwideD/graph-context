import torch
import torch.nn as nn
import torch.nn.functional as F

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, pos):
        ep_cnt = len(pos)
        e_logits = torch.stack([torch.max(logits[st: en, :], dim=0)[0] for st, en in pos]).to(logits)
        e_logits[:, 0] = logits[:ep_cnt, 0]
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        e_logit1 = e_logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(e_logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        e_logit2 = e_logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(e_logit2, dim=-1) * th_label).sum(1)
        '''
        loss2 = []
        for i in range(ep_cnt):
            m_logits = logits[pos[i, 0]:pos[i, 1], :]   
            m_n_mask = n_mask[i:i+1, :].repeat(pos[i, 1]-pos[i, 0], 1)

            m_logits = m_logits - (1 - m_n_mask) * 1e30
            m_logits = m_logits.view(-1)
            m_logits[0] = logits[i, 0]
            th_label = torch.zeros_like(m_logits, dtype = torch.float).to(labels)
            th_label[0] = 1.0
            loss2.append(-(F.log_softmax(m_logits) * th_label).sum())
        ep = torch.zeros_like(e_logit1).to(labels)
        mp = torch.cat([n_mask[i:i+1, :].repeat(pos[i, 1]-pos[i, 0], 1) for i in range(ep_cnt)], dim=0).to(labels)
        m_n_mask = torch.cat([ep, mp], dim=0).to(labels)
        m_n_mask[:ep_cnt, 0] = 1.0
        m_logit2 = logits - (1 - m_n_mask) * 1e30
        th_label = torch.zeros_like(m_logit2, dtype=torch.float).to(labels)
        th_label[:ep_cnt, 0] = 1.0
        loss2 = -(F.log_softmax(m_logit2, dim=-1) * th_label).sum(1)

        loss2 = torch.Tensor(loss2).to(labels)'''
        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1, pos = None):
        e_logits = logits[0: pos[0, 0], :]
        th_logit = e_logits[: , 0].unsqueeze(1)
        output = torch.zeros_like(e_logits).to(logits)
        e_logits = torch.stack([torch.max(logits[st: en, :], dim=0)[0] for st, en in pos]).to(logits)
        mask = (e_logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(e_logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (e_logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output
