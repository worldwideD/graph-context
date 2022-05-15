import torch
import torch.nn as nn
import torch.nn.functional as F

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, e_logits, labels, mplabels, cut):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0
        mplabels[:, 0] = 0.0

        #p_mask = labels + th_label
        n_mask = 1 - mplabels

        # Rank positive classes to TH
        #logit1 = e_logits - (1 - p_mask) * 1e30
        #loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        p1 = torch.sigmoid(e_logits)
        loss1 = -(torch.log(p1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        th_label = torch.zeros_like(mplabels, dtype=torch.float).to(mplabels)
        th_label[:, 0] = 1.0
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        loss2 = torch.split(loss2, cut, dim=0)
        f = lambda x: torch.mean(x, dim=0)
        loss2 = torch.stack(list(map(f, loss2))).to(mplabels)
        
        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1, cut = None):
        logits = logits.float()
        th_logit = logits[:, 0].unsqueeze(1)
        logits = logits - th_logit
        mp_logits = torch.split(logits, cut, dim=0)
        f = lambda x: torch.max(x, dim=0)[0]
        e_logits = torch.stack(list(map(f, mp_logits))).to(logits)

        mask = (e_logits > 0.)
        if num_labels > 0:
            top_v, _ = torch.topk(e_logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (e_logits >= top_v.unsqueeze(1)) & mask
        
        output = torch.zeros_like(e_logits).to(logits)
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output, e_logits
