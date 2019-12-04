import math
import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_add

from .language import InstructionEncoder
from .unet.unet_5_contextual_bneck3 import Unet5ContextualBneck
from .unet.unet_3_contextual_bneck import Unet3ContextualBneck
from .components import position_encoding_1D,Attention,Bottleneck,DenseMlpBlock2


def scatter_logsumexp(src, index, dim=-1, dim_size=None, fill_value=-float("Inf")):
    # Using the logsumexp trick to avoid underflow/overflow, e.g.
    # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
    # First find the max
    max_val,_ = scatter_max(src, index, dim=dim, dim_size=dim_size, fill_value=fill_value)
    # Subtract max from src (via gather function) then exponentiate
    max_gather = max_val.gather(dim,index)
    valid = ~torch.isinf(src) & ~torch.isinf(max_gather)
    src_sub_exp = (src - torch.where(valid, max_gather, torch.zeros_like(src))).exp()
    # Output is max plus log sum of exponentials
    add = scatter_add(src_sub_exp, index, dim=dim, dim_size=max_val.shape[-1])
    valid = ~torch.isinf(max_val) & ~torch.isinf(add)
    max_val = max_val + torch.where(valid, add, torch.ones_like(max_val)).log()
    return max_val


class LingUnet(nn.Module):

    def __init__(self, in_channels, out_channels, embedding_size, norm, activation=nn.ReLU,
              downsample_factor=1, arch=Unet5ContextualBneck):
        super(LingUnet, self).__init__()
        self.unet_posterior = arch(
            in_channels, out_channels, embedding_size,
            hc1=48, hb1=24, hc2=128,
            norm=norm,
        )
        self.act = activation()
        self.downsample = downsample_factor > 1
        if self.downsample:
            self.max_pool = nn.MaxPool2d(downsample_factor, stride=downsample_factor)

    def init_weights(self):
        self.unet_posterior.init_weights()

    def forward(self, semantic_map, h0):
        if self.downsample:
            semantic_map = self.max_pool(semantic_map)
        out = self.unet_posterior(semantic_map, h0)
        if self.act is None:
            return out
        else:
            return self.act(out)


class UnetNavigator(nn.Module):

    def __init__(self, args):
        super(UnetNavigator, self).__init__()
        self.args = args
        self.encoder = InstructionEncoder(args)
        self.lingunet = LingUnet(args.map_depth, 2, args.dec_hidden_size, args.norm_layer, downsample_factor=args.belief_downsample_factor)

    def init_weights(self):
        self.lingunet.init_weights()

    def forward(self, seq, seq_lengths, semantic_map):
        ctx, h0, c0 = self.encoder(seq, seq_lengths)
        out = self.lingunet(semantic_map, h0)
        return out


class MotionModel(torch.nn.Module):
    """ Motion model optionally conditioned on an action embedding """

    def __init__(self, args, action_embedding_size):
        super(MotionModel, self).__init__()
        self.C = args.heading_states
        self.map_channels = args.map_depth
        self.K = args.motion_kernel_size
        self.hidden_size = args.motion_hidden_size
        if self.C == 4:
            seq = torch.arange(-(self.K-1)*0.5,(self.K-1)*0.5+1, device=args.device)
            cell_heading = torch.atan2(seq.unsqueeze(1), seq.unsqueeze(0))
            heading_weights = torch.cat([cell_heading.cos().unsqueeze(0),
                                         cell_heading.sin().unsqueeze(0),
                                        -cell_heading.cos().unsqueeze(0),
                                        -cell_heading.sin().unsqueeze(0)], dim=0).clamp(0,1)
            heading_weights /= heading_weights.sum(dim=0).unsqueeze(0)
            self.heading_weights = heading_weights.reshape(1, 1, self.C, self.K*self.K, 1, 1).log()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dilation = range((self.K-1)//2, 0, -2)
        self.sparse = args.motion_map_sparse
        self.bias = args.motion_map_bias
        self.kernel_params = self.K**2
        self.map_conv = Bottleneck(self.map_channels, self.hidden_size, self.map_channels,
                                            dilation=self.dilation, sparse=self.sparse, bias=self.bias)

        s = args.belief_downsample_factor
        self.max_pool = nn.MaxPool2d(s, stride=s)

        self.pad = nn.ConstantPad2d(self.K//2, -1)
        self.conv1x1 = nn.Conv2d(self.map_channels, self.kernel_params, kernel_size=1, bias=True)
        self.action = DenseMlpBlock2(action_embedding_size, self.hidden_size, self.kernel_params)
        self.im2col = nn.Unfold([self.K, self.K])
        self.dropout = nn.Dropout2d(p=args.motion_dropout_ratio)

    def init_weights(self):
        nn.init.normal_(self.conv1x1.weight)
        self.map_conv.init_weights()

    def propagate_belief(self, log_belief, log_kernel):
        """
        Args:
            log_belief: (N, Cin, H, W) - log probability distribution
            log_kernel: (N, Cin, Cout*K*K, H, W) - log of motion model kernel, normalized over Cout*K*K
        Returns:
            new_log_belief: (N, Cout, H, W)
        """

        N = log_belief.shape[0]
        Cin = log_belief.shape[1]
        H = log_belief.shape[2]
        W = log_belief.shape[3]
        Cout = int(log_kernel.shape[2]/(self.K*self.K))
        # Transposed convolution of spatially-varying kernel with log belief
        # Standard convolution can't be used because the time-varying kernel can't be easily normalized in a standard conv,
        # but this is easy to do in a transposed conv that is distributing values to the next layer
        # (N, Cin, 1, H, W) + (N, Cin, Cout*K*K, H, W) -> (N, Cin, Cout*K*K, H, W)
        col2im = (log_belief.unsqueeze(2) + log_kernel).reshape(N*Cin,-1)

        # We need to sum over outputs that correspond to the same position in the map
        # Set up img_ix which will map elements in col2im back to map space
        # Pytorch fold can't be used because it sums overlapping values, we need the logsumexp operation
        img_ix = torch.arange(Cout*H*W, device=log_belief.device, dtype=log_belief.dtype).reshape(1, Cout, H, W)
        img_ix = self.im2col(self.pad(img_ix)).expand(N*Cin, -1, -1, -1).reshape(N*Cin,-1)
        valid_ix = img_ix >= 0
        new_belief = scatter_logsumexp(col2im[valid_ix].reshape(N,-1), img_ix[valid_ix].reshape(N,-1).long()).reshape(N, -1, H, W)
        return new_belief


    def forward(self, log_belief, semantic_map, action):
        """ Propagate log belief through motion kernels
        Args:
            log_belief (torch.FloatTensor): Log of belief
            semantic_map (torch.FloatTensor): Semantic map
            h: optional language embedding
        Returns:
            log_belief (torch.FloatTensor): Updated log of belief
        Shape:
            Input:
                log_belief: (N, 1, map_y, map_x)
                semantic_map: (N, map_depth, map_y, map_x)
            Output:
                log_belief: (N, Cout, map_y, map_x)
        """

        N = log_belief.shape[0]
        H = log_belief.shape[2]
        W = log_belief.shape[3]

        # Predict kernel params based on the semantic map and normalize over K*K to preserve probability mass
        motion_log_kernel = self.log_softmax(self.conv1x1(self.map_conv(self.max_pool(self.dropout(semantic_map))))).unsqueeze(1) # (N, 1, K*K, H, W)

        # Predict a kernel based on language and normalize over K*K
        language_log_kernel = self.log_softmax(self.action(action)).unsqueeze(1).unsqueeze(-1).unsqueeze(-1) # (N, 1, K*K, 1, 1)

        if self.C > 1:
            assert self.C==4, 'Can only rotate motion model with 4 heading stages'
            # Update motion kernel to distribute output belief over heading bins
            # This is deterministic since we assume robot only turns and moves forward - so heading is known from displacement
            # (N, Cin, Cout*K*K, H, W)
            motion_log_kernel = motion_log_kernel.unsqueeze(2) + self.heading_weights.expand(N, self.C, -1, -1, H, W)
            motion_log_kernel = motion_log_kernel.reshape(N, self.C, self.C*self.K*self.K, H, W)

            # Rotate the language part of the kernel so the prediction is in an egocentric frame
            lk = language_log_kernel.reshape(N, self.K, self.K)
            language_log_kernel = torch.cat([lk.unsqueeze(1),
                                             lk.rot90(1,[2, 1]).unsqueeze(1), # rotate x to y
                                             lk.rot90(2,[2, 1]).unsqueeze(1),
                                             lk.rot90(3,[2, 1]).unsqueeze(1)], dim=1).reshape(N, self.C, self.K*self.K, 1, 1)

            language_log_kernel = language_log_kernel.unsqueeze(2) + self.heading_weights.expand(N, self.C, -1, -1, 1, 1)
            language_log_kernel = language_log_kernel.reshape(N, self.C, self.C*self.K*self.K, 1, 1)

        log_kernel = motion_log_kernel + language_log_kernel
        log_kernel = log_kernel - log_kernel.logsumexp(dim=2, keepdim=True) # renormalize

        new_belief = self.propagate_belief(log_belief, log_kernel)
        return new_belief


class ObservationModel(nn.Module):
    """ Observation model conditioned on language (observation) """
    def __init__(self, args, obs_embedding_size):
        super(ObservationModel, self).__init__()
        out_channels = args.heading_states
        self.lingunet = LingUnet(args.map_depth, out_channels, obs_embedding_size,
                                 args.norm_layer, activation=nn.LogSigmoid,
                                 downsample_factor=args.belief_downsample_factor,
                                 arch=Unet3ContextualBneck)
        self.min_obs_likelihood = math.log(args.min_obs_likelihood)

    def init_weights(self):
        self.lingunet.init_weights()

    def forward(self, obs, semantic_map):
        out = self.lingunet(semantic_map, obs)
        out = torch.clamp(out, min=self.min_obs_likelihood)
        return out


class ActionObservationDecoder(nn.Module):
    """ Decodes language to a series of latent actions and observations """
    def __init__(self, args):
        super(ActionObservationDecoder, self).__init__()
        # Encoder
        num_directions = 2 if args.bidirectional else 1
        enc_hidden_size = args.enc_hidden_size*num_directions

        self.encoder = InstructionEncoder(args)
        # Decoder
        self.num_embeddings = args.max_steps*5
        self.embed_size = args.action_embedding_size
        pos = position_encoding_1D(self.num_embeddings, self.embed_size)
        self.pos_encoding = nn.Parameter(pos, requires_grad=False)
        self.pos_embedding = nn.Embedding(self.num_embeddings, self.embed_size,
                                _weight=self.pos_encoding)
        dec_input_size = args.action_embedding_size + enc_hidden_size
        self.decoder = nn.LSTMCell(dec_input_size, args.dec_hidden_size)
        self.ones = torch.ones(args.batch_size, device=args.device).long()
        # Soft attention
        self.act_att = Attention(args.dec_hidden_size, enc_hidden_size, args.att_hidden_size)
        self.obs_att = Attention(args.dec_hidden_size, enc_hidden_size, args.att_hidden_size)

    def init_weights(self):
        pass

    def forward(self, t, seq, seq_mask, seq_lens, state=None):
        if state is None:
            enc_ctx, h, c = self.encoder(seq, seq_lens)
            # First time step, input average context vector
            prev = enc_ctx.mean(dim=1)
        else:
            enc_ctx, h, c, prev = state
        embed = torch.cat([self.pos_embedding(t*self.ones), prev], dim=1)
        h, c = self.decoder(embed, (h, c))
        obs, obs_att_weights = self.act_att(h, enc_ctx, seq_mask)
        act, act_att_weights = self.obs_att(h, enc_ctx, seq_mask)
        act1 = torch.cat([act,h], dim=1)
        obs1 = torch.cat([obs,h], dim=1)
        return act1, obs1, (enc_ctx, h, c, act+obs), act_att_weights, obs_att_weights

    def detach_state(self, state):
        enc_ctx, h, c, prev = state
        h.detach_()
        c.detach_()
        prev.detach_()
        return (enc_ctx.detach(), h, c, prev)


class Filter(nn.Module):
    """ Bayes filter for ideal agent """
    def __init__(self, args):
        super(Filter, self).__init__()
        self.args = args
        self.decoder = ActionObservationDecoder(args)
        num_directions = 2 if args.bidirectional else 1
        obs_size = args.enc_hidden_size*num_directions + args.dec_hidden_size
        action_size = obs_size
        self.motion = MotionModel(args, action_size)
        self.obs = ObservationModel(args, obs_size)

    def init_weights(self):
        self.decoder.init_weights()
        self.motion.init_weights()
        self.obs.init_weights()

    def forward(self, t, seq, seq_mask, seq_lens, log_belief, semantic_map, state=None, gt_log_belief=None):
        act,obs,state,act_att_weights,obs_att_weights = self.decoder(t, seq, seq_mask, seq_lens, state=state)
        new_belief = self.motion(log_belief, semantic_map, act)
        if type(gt_log_belief) != type(None):
            new_gt_belief = self.motion(gt_log_belief, semantic_map, act)
        else:
            new_gt_belief = None
        obs_likelihood = self.obs(obs, semantic_map)
        return new_belief, obs_likelihood, state, act_att_weights, obs_att_weights, new_gt_belief

    def detach_state(self, state):
        return self.decoder.detach_state(state)
