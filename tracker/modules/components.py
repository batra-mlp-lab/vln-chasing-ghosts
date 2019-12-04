import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SparseConv2d(nn.Module):
    """ Conv2d that normalizes for sparse inputs in the manner of
        https://arxiv.org/pdf/1708.06500.pdf
    Shape:
        Input:
            input (N, C, H, W)
            mask (N, 1, H, W) indicating valid inputs
        Output: (N, C, H, W) (same shape as input)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SparseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                 kernel_size, stride=stride, padding=padding,
                 dilation=dilation, groups=groups, bias=bias)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups,
                 bias=False)
        for param in self.mask_conv.parameters():
            param.data.fill_(1)
            param.requires_grad = False

    def forward(self, input, mask=None):
        if mask is None:
            N,C,H,W = input.shape
            mask = torch.any((input.reshape(N,C,-1)!=0),dim=1).reshape(N,1,H,W).type(input.dtype)
        upmask = self.mask_conv(mask)
        out = self.conv(input) * mask / (upmask + 1e-10)
        return out


class MultiDilatedConv2d(nn.Module):
    """ Conv2d with multiple dilations
    Shape:
        Input: (N, C, H, W)
        Output: (N, C, H, W) (same shape as input)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=[0], dilation=[1], groups=1, bias=True):
        super(MultiDilatedConv2d, self).__init__()
        self.conv = nn.ModuleList()
        for p,d in zip(padding,dilation):
            self.conv.append(nn.Conv2d(in_channels, out_channels//len(dilation),
                     kernel_size, stride=stride, padding=p,
                     dilation=d, groups=groups, bias=bias))

    def forward(self, input):
        out = torch.cat([conv(input) for conv in self.conv], dim=1)
        return out


class StatefulDropout(nn.Module):
    """ Dropout that will maintain the dropout mask until new_mask is called.
    Args:
        p (float, optional): probability of an element to be zero-ed.
        spatial (bool, optional): If true performs spatial dropout, by randomly
                zeroing out entire channels as described in the paper
                `Efficient Object Localization Using Convolutional Networks`.
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W) (same shape as input)
    """
    def __init__(self, device, p=0.5, spatial=True):
        super(StatefulDropout, self).__init__()
        self.device = device
        self.p = p
        self.spatial = spatial
        self.new_mask()

    def new_mask(self):
        self.needs_new_mask = True

    def forward(self, input):
        if self.training:
            if self.needs_new_mask:
                if self.spatial:
                    probs = torch.ones(input.shape[:2]) * (1.0-self.p)
                    self.mask = torch.bernoulli(probs).unsqueeze(2).unsqueeze(2).to(self.device)
                else:
                    probs = torch.ones(input.shape[0], input.shape[2], input.shape[3]) * (1.0-self.p)
                    self.mask = torch.bernoulli(probs).unsqueeze(1).to(self.device)
                self.needs_new_mask = False
            out = input * self.mask
        else:
            out = input / (1.0-self.p)
        return out


class Attention(nn.Module):
    """Generic Attention module"""

    def __init__(self, input_dim, candidate_dim, hidden_size, output_logits=False):
        """Initialize layer
        Args:
            output_logits (bool): If False, returns unnormalized attention weights, otherwise
                    attended features and attention weights are returned.
        """
        super(Attention, self).__init__()
        self.linear_h = nn.Linear(input_dim, hidden_size, bias=False)
        self.linear_c = nn.Linear(candidate_dim, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.output_logits = output_logits

    def forward(self, h, candidates, mask=None):
        """ Calculates attention
        Args:
            h (torch.cuda.FloatTensor): Hidden state input
            candidates (torch.cuda.FloatTensor): Attention candidates
            mask (torch.cuda.ByteTensor): Indices to be masked. Default=None
        Returns:
            weights (torch.cuda.FloatTensor): attention weights
            features (torch.cuda.FloatTensor): attended features
        Shape:
            Input:
                h: (batch_size, input_dim)
                candidates: (batch_size, n_items, candidate_dim)
                mask: (batch_size, n_items)
            Output:
                weights: (batch_size, n_items)
                features: (batch_size, candidate_dim)
        """
        # shape: (batch_size, n_items, hidden_size)
        values = self.linear_c(candidates)
        # shape: (batch_size, n_items, hidden_size)
        keys = (self.linear_h(h).unsqueeze(1).expand_as(values))
        # shape: (batch_size, n_items)
        affinities = self.linear_out(self.tanh(keys + values)).view(keys.shape[0], keys.shape[1])
        if mask is not None:
            # -Inf masking prior to the softmax
            affinities.masked_fill_(mask, -float("inf"))
        if self.output_logits:
            return affinities
        else:
            # shape: (batch_size, n_items)
            weights = self.softmax(affinities)
            # (batch_size, 1, n_items) * (batch_size, n_items, candidate_dim)
            features = torch.bmm(weights.unsqueeze(1), candidates).squeeze(1)
            return features, weights


class DenseMlpBlock2(nn.Module):
    """ Attributed to Cornell Language in Context Lab https://github.com/clic-lab/drif """
    def __init__(self, in_size, hidden_size, out_size):
        super(DenseMlpBlock2, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(in_size + hidden_size, out_size)
        self.act1 = nn.LeakyReLU()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        torch.nn.init.kaiming_normal_(self.linear1.bias)
        torch.nn.init.kaiming_normal_(self.linear2.bias)

    def forward(self, input):
        x1 = self.act1(self.linear1(input))
        x1_cat = torch.cat([input, x1], dim=1)
        x2 = self.linear2(x1_cat)
        return x2


class NormalEstimator(nn.Module):
    ''' Very simple normal estimation from a depth map '''

    def __init__(self, device):
        super(NormalEstimator, self).__init__()
        # Sobel filters
        self.Gx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='border')
        kernel = np.array([[-1, 0, 1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        self.Gx.weight = nn.Parameter(torch.from_numpy(kernel).to(device).unsqueeze(0).unsqueeze(0), requires_grad=False)

        self.Gy = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='border')
        kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
        self.Gy.weight = nn.Parameter(torch.from_numpy(kernel).to(device).unsqueeze(0).unsqueeze(0), requires_grad=False)

        self.ones = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='border')
        self.ones.weight = nn.Parameter(torch.ones_like(self.ones.weight).to(device), requires_grad=False)

    def forward(self, depth, vfov_radians):
        image_height = depth.shape[-2]
        f = image_height / (2.0*math.tan(vfov_radians/2.0))
        scale = 8.0*depth/f
        grad_x = self.Gx(depth)/scale
        grad_y = self.Gy(depth)/scale

        normal = torch.cat([-grad_x, -grad_y, torch.ones_like(grad_x)], dim=1)
        norm = torch.norm(normal, p=2, dim=1, keepdim=True)
        normal /= norm

        # Zero out invalid, assuming zero depth values are missing
        invalid = self.ones((depth==0).float()) != 0
        normal[invalid.expand_as(normal)] = 0
        return normal


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.padding = kernel_size//2
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)


def position_encoding_1D(num_embeddings, embedding_dim):
    """Init a symmetrical sinusoid 1D position encoding table

    Args:
        num_embeddings (int)
        embedding_dim (int)

    Returns:
        (torch.cuda.FloatTensor): Embedding

    Shape:
        Output:
            (num_embeddings, embedding_dim)
    """
    if num_embeddings % 2 == 0:
        indices = range(-num_embeddings // 2, num_embeddings // 2 + 1)
        indices.remove(0)
    else:
        indices = range(-num_embeddings // 2 + 1, num_embeddings // 2 + 1)

    # shape: (num_embeddings, embedding_dim)
    position_enc = np.array(
        [
            [
                pos / np.power(10000, 2 * (j // 2) / embedding_dim)
                for j in range(embedding_dim)
            ]
            for pos in indices
        ]
    )

    # apply sin on 0th, 2nd, 4th...embedding_dim
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    # apply cos on 1st, 3rd, 5th...embedding_dim
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

    return torch.from_numpy(position_enc).type(torch.cuda.FloatTensor)


def position_encoding_2D(num_y_embeddings, num_x_embeddings, embedding_dim):
    """Init a symmetrical sinusoid @D position encoding table

    Args:
        num_y_embeddings (int)
        num_x_embeddings (int)
        embedding_dim (int)

    Returns:
        encoding (torch.cuda.FloatTensor): Embedding

    Shape:
        Output:
            embedding: (num_y_embeddings * num_x_embeddings, embedding_dim)
    """

    x_dim = embedding_dim // 2
    y_dim = embedding_dim - x_dim

    # shape: (num_x_embeddings, x_dim)
    x_embed = position_encoding_1D(num_x_embeddings, x_dim)
    # shape: (num_y_embeddings, num_x_embeddings, x_dim)
    x_embed = x_embed.unsqueeze(0).expand(num_y_embeddings, num_x_embeddings, x_dim)
    # shape: (num_y_embeddings, y_dim)
    y_embed = position_encoding_1D(num_y_embeddings, y_dim)
    # shape: (num_y_embeddings, num_x_embeddings, x_dim)
    y_embed = y_embed.unsqueeze(1).expand(num_y_embeddings, num_x_embeddings, y_dim)

    # shape: (num_y_embeddings * num_x_embeddings, embedding_dim)
    encoding = torch.cat([y_embed, x_embed], dim=2).reshape(-1, embedding_dim)
    return encoding


class Bottleneck(nn.Module):
    """ Closely related to residual net Bottleneck. Optionally conditions on a language embedding as well """

    def __init__(self, inplanes, planes, outplanes, embedding_size=0, stride=1, dilation=1, sparse=True, bias=True):
        super(Bottleneck, self).__init__()
        if sparse:
            conv_layer = SparseConv2d
        elif hasattr(dilation, '__iter__'):
            conv_layer = MultiDilatedConv2d
        else:
            conv_layer = nn.Conv2d
        self.conv1 = conv_layer(inplanes, planes, kernel_size=1, stride=stride, bias=bias)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=bias, dilation=dilation)
        self.conv3 = conv_layer(planes+embedding_size, outplanes, kernel_size=1, stride=stride, bias=bias)
        self.relu = nn.ReLU()
        self.embedding_size = embedding_size
        self.stride = stride

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not (m.out_channels == 1 and m.in_channels == 1):
                    # Don't init mask in SparseConv2d
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, embedding=None):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        if self.embedding_size > 0:
            out = torch.cat([embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.shape[-2], out.shape[-1]), out], dim=1)
        out = self.conv3(out)

        if out.shape[-1] == identity.shape[-1]:
            out += identity
        out = self.relu(out)

        return out
