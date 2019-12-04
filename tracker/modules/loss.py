import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input, target, mask=None):
        if mask is None:
            self.loss = self.criterion(input, target)
        else:
            while len(mask.shape) < len(input.shape):
                mask = mask.unsqueeze(-1)
            self.loss = self.criterion(input*mask, target*mask)
        return self.loss


class LogBeliefLoss(nn.Module):
    def __init__(self):
        super(LogBeliefLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')

    def forward(self, predicted_logprob, target_prob, valid=None):
        """ Calculate KL divergence loss, excluding zero prob predictions as necessary """
        N = target_prob.shape[0]
        if valid is not None:
            if len(predicted_logprob.shape) == 3:
                valid = (~torch.isinf(predicted_logprob) & valid.view(N, 1, 1))
            else:
                valid = (~torch.isinf(predicted_logprob) & valid.view(N, 1, 1, 1))
        else:
            valid = ~torch.isinf(predicted_logprob)
        loss = self.criterion(predicted_logprob, target_prob)[valid].sum()
        return loss
