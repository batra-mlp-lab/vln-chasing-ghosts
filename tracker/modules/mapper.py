import math
import torch
import torch.nn as nn
import torchgeometry as tgm

from .cnn import DenserResNet34
from .feature_projection import FeatureProjection
from .components import StatefulDropout


class Mapper(nn.Module):
    '''A simple non-dynamic Mapper module based on convolutional GRU updates'''

    def __init__(self, args):
        super(Mapper, self).__init__()

        self.cnn = DenserResNet34(finetune=args.finetune_cnn)
        self.feature_proj = FeatureProjection(args.vfov, args.batch_size, self.cnn, args.cnn_feature_map_height,
                args.cnn_feature_map_width, args.map_range_y, args.map_range_x, args.gridcellsize, args.z_clip_threshold,
                args.add_depth_to_map_features, args.depth_scaling, device=args.device)
        self.args = args
        self.padding = (args.kernel_size // 2, args.kernel_size // 2)
        assert args.map_range_y % 2 == 0 and args.map_range_x % 2 == 0,\
                 'map_range_x and _y should be even'
        assert args.map_range_y % args.belief_downsample_factor == 0 and args.map_range_x % args.belief_downsample_factor == 0,\
                 'belief_downsample_factor should evenly divide the map_range'
        self.belief_range_y = int(args.map_range_y / args.belief_downsample_factor)
        self.belief_range_x = int(args.map_range_x / args.belief_downsample_factor)

        self.x2h = nn.Conv2d(self.args.map_feature_size, 3 * args.map_depth, args.kernel_size,
                              padding=self.padding, bias=True)
        self.h2h = nn.Conv2d(args.map_depth, 3 * args.map_depth, args.kernel_size,
                              padding=self.padding, bias=True)
        self.x2h_drop = StatefulDropout(args.device, p=args.mapper_dropout_ratio)
        self.h2h_drop = StatefulDropout(args.device, p=args.mapper_dropout_ratio)

        self.mask_conv = nn.Conv2d(1, 1, args.kernel_size, padding=self.padding, bias=False)
        for param in self.mask_conv.parameters():
            param.data.fill_(1)
            param.requires_grad = False


    def init_map(self, xyzhe):
        ''' Initialize map with state defined by x,h,z,heading,elevation '''
        batch_size = xyzhe.shape[0]
        assert batch_size == self.args.batch_size
        # Map will be xy centered at the state location of the first observation
        self.map_center = torch.zeros_like(xyzhe)
        self.map_center[:,:2] = -xyzhe[:,:2]
        # Build 3D transformation matrices for world-to-map
        angles = torch.zeros(batch_size,3, device=self.args.device)
        self.T_s = tgm.angle_axis_to_rotation_matrix(angles) # Nx4x4
        self.T_s[:,0,3] = self.map_center[:,0]
        self.T_s[:,1,3] = self.map_center[:,1]
        # Setup dropout masks, which will be maintained for the entire episode
        self.x2h_drop.new_mask()
        self.h2h_drop.new_mask()
        # Return empty map and mask
        new_map = torch.zeros(batch_size, self.args.map_depth, self.args.map_range_y,
                            self.args.map_range_x, device=self.args.device)
        new_mask = torch.zeros(batch_size, 1, self.args.map_range_y,
                            self.args.map_range_x, device=self.args.device)
        return new_map, new_mask


    def forward(self, rgb, depth, xyzhe, prev_map, prev_mask):
        """ Update map for rgb and depth images at each state. RGB and Depth may contain
            more than one observation per map to be processed simultaneously,
            e.g. for projecting a pano image to the map. In this case every
            batch size'th item projects to the same map.
        Args:
            rgb (torch.cuda.FloatTensor): RGB image with values in [0, 255]
            depth (torch.cuda.FloatTensor): Depth image
            xyzhe (torch.cuda.FloatTensor): Simulator state (x,y,z,heading,elevation)
            prev_map (torch.cuda.FloatTensor): Previous spatial map
            prev_mask (torch.cuda.FloatTensor): Previous map mask
        Returns:
            spatial_map (torch.cuda.FloatTensor): Projected features onto 2D grid based on
                                                  mappings computed
            mask (torch.cuda.FloatTensor): Tensor of 0s and 1s where 1 tells that a non-zero
                                           feature is present at that (i,j) coordinate
            features_to_map (torch.cuda.LongTensor): World (x,y) coordinates of features
                        discretized in gridcellsize and cropped to (output_height, output_width)
                        and use offset as origin.
        Shape:
            Input:
                rgb: (N*obs_per_map, 3, height, width)
                depth: (N*obs_per_map, 1, height, width)
                T: (N*obs_per_map, 4, 4)
                prev_map: (N, feature_size, output_height, output_width)
                prev_mask: (N, output_height, output_width)
            Output:
                new_map: (N, feature_size, output_height, output_width)
                mask: (N, 1, output_height, output_width)
                features_to_map: (N*obs_per_map, features_height, features_width, 2)
        """

        obs_per_map = int(rgb.shape[0]/self.args.batch_size)
        T_w = self._transform3D(xyzhe)
        T = torch.bmm(self.T_s.repeat(obs_per_map,1,1), T_w) # Adjust coordinates for map centering and map orientation
        proj, mask, features_to_map = self.feature_proj(rgb, depth, T, obs_per_map)
        conv_proj = proj.permute(0, 3, 1, 2)  # (N,C,H,W)
        obs_mask = mask.unsqueeze(1)  # (N,1,H,W)

        # GRU operation
        gate_x = self.x2h(conv_proj) * obs_mask
        gate_h = self.h2h(prev_map) * prev_mask

        gate_x = self.x2h_drop(gate_x)
        gate_h = self.h2h_drop(gate_h)

        i_r, i_z, i_n = gate_x.chunk(3, 1)
        h_r, h_z, h_n = gate_h.chunk(3, 1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + (r * h_n))

        # z should be 1 if we have no new observation
        z = torch.where((obs_mask==0).expand(-1,self.args.map_depth,-1,-1),
                            torch.tensor(1.0, device=self.args.device), z)

        new_map = z * prev_map + (1-z) * n
        new_mask = ((prev_mask + obs_mask)>0).float()

        return new_map,new_mask,features_to_map


    def _transform3D(self, xyzhe):
        """ Return (N, 4, 4) transformation matrices from (N,5) x,y,z,heading,elevation """

        theta_x = -math.pi/2.0 + xyzhe[:,4] # elevation
        cx = torch.cos(theta_x)
        sx = torch.sin(theta_x)

        theta_z = -xyzhe[:,3] # heading
        cz = torch.cos(theta_z)
        sz = torch.sin(theta_z)

        T = torch.zeros(xyzhe.shape[0], 4, 4, device=self.args.device)
        T[:,0,0] = cz
        T[:,0,1] = -sz*cx
        T[:,0,2] = sz*sx
        T[:,0,3] = xyzhe[:,0] # x

        T[:,1,0] = sz
        T[:,1,1] = cz*cx
        T[:,1,2] = -cz*sx
        T[:,1,3] = xyzhe[:,1] # y

        T[:,2,0] = 0
        T[:,2,1] = sx
        T[:,2,2] = cx
        T[:,2,3] = xyzhe[:,2] # z

        T[:,3,3] = 1
        return T


    def belief_map(self, xyzhe, sigma):
        """ Return the belief map for the given simulator heading and location.
        Args:
            xyzhe: agent pose (x,y,z,heading,elevation) in world coordinates
            sigma: standard dev of belief
        Shape:
            Inputs:
                xyzhe: (N, 5)
                valid: (N)
            Outputs:
                belief: (N, belief_heading_bins, belief_range_y, belief_range_x)
                idx: [N,C,Y,X]
        """
        sigma = sigma/self.args.gridcellsize
        N = xyzhe.shape[0]
        Yr = self.belief_range_y
        Xr = self.belief_range_x
        heat = torch.zeros(N, Yr, Xr, device=self.args.device)
        # Convert world coordinates to map grid coordinates and then belief grid coordinates
        mu = (xyzhe[:,:2] + self.map_center[:,:2])/(self.args.gridcellsize*self.args.belief_downsample_factor)
        heat = self._gaussian(mu, Xr, Yr, sigma)
        if self.args.heading_states == 1:
            belief = heat.unsqueeze(1)
        else:
            # Convert Matterport3D Simulator heading to standard definition rel to x and y axes
            heading = 0.5*math.pi-xyzhe[:,3]
            # Normalize heading between 0 and 2pi
            heading.fmod_(2*math.pi)
            heading = (heading + 2*math.pi).fmod(2*math.pi)
            # With 4 heading states the channels will represent +x, +y, -x, -y
            bin_width = 2*math.pi/self.args.heading_states
            default_headings = bin_width*torch.arange(start=0,end=self.args.heading_states, device=self.args.device, dtype=xyzhe.dtype)
            bin_weights = (heading.unsqueeze(1) - default_headings.unsqueeze(0)).cos().clamp(0,1)
            # Weight each heading bin and normalize
            bin_weights = bin_weights / bin_weights.sum(dim=1).unsqueeze(1)
            belief = heat.unsqueeze(1) * bin_weights.unsqueeze(2).unsqueeze(2)
        # Renormalize just in case, mainly needed when sigma is very small
        belief /= belief.reshape(N,-1).sum(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        return belief


    def _gaussian(self, mu, x_range, y_range, sigma):
        """ Generate gaussians centered at mu of the given x and y range with the origin in the centre of the output """
        x_mu = (mu[:,0] + x_range//2).unsqueeze(1).unsqueeze(1)
        y_mu = (mu[:,1] + y_range//2).unsqueeze(1).unsqueeze(1)
        # Subtract half a grid cell if the map has an even number of cells, since the origin/center of map
        # should therefore be on the boundary of two cells (not in the middle of a cell)
        if x_range % 2 == 0:
            x_mu -= 0.5
        if y_range % 2 == 0:
            y_mu -= 0.5
        # Generate bivariate Gaussians centered at position mu
        x = torch.arange(start=0,end=x_range, device=mu.device, dtype=mu.dtype).unsqueeze(0).unsqueeze(0)
        y = torch.arange(start=0,end=y_range, device=mu.device, dtype=mu.dtype).unsqueeze(1).unsqueeze(0)
        gauss = torch.exp(-0.5 * (((x - x_mu)/sigma)**2 + ((y - y_mu)/sigma)**2)) / (2.0 * math.pi * sigma**2)
        return gauss


    def heatmap(self, xy, heatmap_sigma):
        """ Generate heatmaps at the provided simulator positions
        Inputs:
            xy: (N, 2) or (K, N, 2)
        Outputs:
            heat: (N, output_height, output_width)
        """
        sigma = heatmap_sigma/self.args.gridcellsize
        N = xy.shape[-2]
        Xr = self.args.map_range_x
        Yr = self.args.map_range_y
        if len(xy.shape) == 2:
            xy = xy.unsqueeze(0)
        heat = torch.zeros(N, Yr, Xr, device=self.args.device)
        for k,xy_ in enumerate(xy):
            # Convert world coordinates to map grid coordinates
            mu = (xy_ + self.map_center[:,:2])/self.args.gridcellsize
            heat = torch.max(heat, self._gaussian(mu, Xr, Yr, sigma))
        return heat
