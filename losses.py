import torch
import torch.nn as nn
import torch.nn.functional as F

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, pos):
        e_logits = torch.stack([torch.max(logits[st: en, :], dim=0)[0] for st, en in pos]).to(logits)
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
        #m_n_mask = torch.stack([torch.ones(pos[i, 1]-pos[i, 0], ) * n_mask[i] for i in range(len)], dim=0).to(labels)
        ep_cnt = len(pos)
        m_n_mask = torch.cat([n_mask[i:i+1, :].repeat(pos[i, 1]-pos[i, 0], 1) for i in range(ep_cnt)]).to(labels)
        m_logit2 = logits[pos[0, 0]:, :] - (1 - m_n_mask) * 1e30
        th_label = torch.zeros_like(m_logit2, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        loss2 = -(F.log_softmax(m_logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        #loss = loss1 + loss2
        #loss = loss.mean()
        loss = loss1.mean() + loss2.mean()
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
