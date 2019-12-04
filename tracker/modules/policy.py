import torch
import torch.nn as nn
import torch.nn.functional as F



class ReactivePolicy(nn.Module):

    def __init__(self, args):
        super(ReactivePolicy, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.repeats = 10
        self.feature_size = 6 + args.max_steps
        self.input_dim = self.repeats * self.feature_size
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, args.policy_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.policy_hidden_size, 1)
        )

    def init_weights(self):
        pass

    def forward(self, features):
        """ Generates a distribution over nodes in the navigation graph
            representing next action
        """
        # drop batch_index from start and duplicate some times
        repeated_features = features[:,1:].repeat(1,self.repeats)
        logits = self.encoder(repeated_features)

        # Reshape and apply softmax
        batch_indices = features[:,0]
        longest_item = batch_indices.mode()[0]
        max_len = (batch_indices==longest_item).sum().item()
        log_prob = torch.ones(self.batch_size, max_len, device=self.args.device) * -float("inf")
        for n in range(self.batch_size):
            ix = batch_indices == n
            log_prob[n,:ix.sum()] = F.log_softmax(logits[ix].reshape(-1), dim=0)
        return log_prob
