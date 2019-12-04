import torch
import torch.nn as nn
from torchvision import models
from torch_scatter import scatter_max
import numpy as np
import math
from .components import NormalEstimator, GaussianSmoothing


def compute_intrinsic_matrix(width, height, vfov):
    f = height / (2.0*math.tan(vfov/2.0))
    cx = width / 2.0
    cy = height / 2.0
    K = torch.Tensor([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    return K


class Conv1x1Block(nn.Module):
    """ Simple 1x1 convolutions """
    def __init__(self, in_size, hidden_size, out_size):
        super(Conv1x1Block, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(in_size, hidden_size, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_size + hidden_size, out_size, kernel_size=1, stride=1, bias=True)
        self.act1 = nn.LeakyReLU()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv1.bias)
        torch.nn.init.kaiming_normal_(self.conv2.bias)

    def forward(self, input):
        x1 = self.act1(self.conv1(input))
        x1_cat = torch.cat([input, x1], dim=1)
        x2 = self.conv2(x1_cat)
        return x2


class FeatureProjection(nn.Module):
    """Extracts CNN features from an RGB-D image and projects them onto a ground plane map.
       The origin is always at the center of the map.
    """

    def __init__(
        self,
        vfov,
        batch_size,
        cnn,
        feature_map_height,
        feature_map_width,
        output_height,
        output_width,
        gridcellsize,
        z_clip_threshold,
        add_depth_to_features,
        depth_scaling,
        device=torch.device("cuda"),
    ):
        """Init function

        Args:
            vfov (float): Vertical Field of View
            batch_size (float)
            cnn (torch.nn.Module): CNN model to be used for computing features from RGB image
            feature_map_height (int): height of features produced by CNN
            feature_map_width (int): width of features produced by CNN
            output_height (int): Height of the spatial map to be produced
            output_width (int): Width of the spatial map to be produced
            gridcellsize (float): How many metres does 1 pixel of spatial map represents
            z_clip_threshold (float): Consider pixels only lying within the z threshold above the
                                      camera (in metres)
            device (torch.device, optional): Defaults to torch.device('cuda').
        """

        super(FeatureProjection, self).__init__()

        self.batch_size = batch_size
        self.vfov = vfov
        self.fmh = feature_map_height
        self.fmw = feature_map_width
        self.output_height = output_height
        self.output_width = output_width
        self.gridcellsize = gridcellsize
        self.z_clip_threshold = z_clip_threshold
        self.depth_scaling = depth_scaling
        self.add_depth_to_features = add_depth_to_features
        if self.add_depth_to_features: 
            self.gauss = GaussianSmoothing(1, 5, 1)
            self.normal = NormalEstimator(device)
            # Also add an mlp
            feat_size = cnn.feature_size()
            self.conv1x1 = Conv1x1Block(feat_size+4, feat_size+4, feat_size)
        self.cnn = cnn
        self.device = device

        self.x_scale, self.y_scale, self.ones = self.compute_scaling_params(
            batch_size, feature_map_height, feature_map_width
        )

        self.avgpool = nn.AdaptiveAvgPool2d((feature_map_height, feature_map_width))

    def compute_scaling_params(self, batch_size, image_height, image_width):
        """ Precomputes tensors for calculating depth to point cloud """
        # (float tensor N,3,3) : Camera intrinsics matrix
        K = compute_intrinsic_matrix(image_width, image_height, self.vfov)
        K = K.to(device=self.device).unsqueeze(0)
        K = K.expand(batch_size, 3, 3)

        fx = K[:, 0, 0].unsqueeze(1).unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1).unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1).unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1).unsqueeze(1)

        x_rows = torch.arange(start=0, end=image_width, device=self.device)
        x_rows = x_rows.unsqueeze(0)
        x_rows = x_rows.repeat((image_height, 1))
        x_rows = x_rows.unsqueeze(0)
        x_rows = x_rows.repeat((batch_size, 1, 1))
        x_rows = x_rows.float()

        y_cols = torch.arange(start=0, end=image_height, device=self.device)
        y_cols = y_cols.unsqueeze(1)
        y_cols = y_cols.repeat((1, image_width))
        y_cols = y_cols.unsqueeze(0)
        y_cols = y_cols.repeat((batch_size, 1, 1))
        y_cols = y_cols.float()

        # 0.5 is so points are projected through the center of pixels
        x_scale = (x_rows + 0.5 - cx) / fx
        y_scale = (y_cols + 0.5 - cy) / fy
        ones = (
            torch.ones((batch_size, image_height, image_width), device=self.device)
            .unsqueeze(3)
            .float()
        )
        return x_scale, y_scale, ones

    def masked_nonzero_pooling(self, img, mask_input=None):
        """
        Avgpool over only non-zero values of the tensor

        Args:
            img (torch.FloatTensor)

        Returns:
            pooled_img (torch.FloatTensor)

        Shape:
            Input:
                img: (batch_size, feature_size, height, width)
            Output:
                pooled_img: (batch_size, feature_size, feature_map_height, feature_map_width)

        Logic:
            MaskedAvgPool = AvgPool(img) / AvgPool(img != 0)
            Denominator for both average pools cancels out.
        """
        if type(mask_input) == type(None):
            mask_input = img
        pooled_img = self.avgpool(img)
        pooled_mask = self.avgpool((mask_input!=0).float())
        nonzero_mask = torch.where(pooled_mask == 0, torch.ones_like(pooled_mask)*1e-8, pooled_mask)
        pooled_img = torch.div(pooled_img, nonzero_mask)
        return pooled_img

    def point_cloud(self, depth, depth_scaling):
        """
        Converts image pixels to 3D pointcloud in camera reference using depth values.

        Args:
            depth (torch.FloatTensor): (batch_size, height, width)
            depth_scaling (float): divisor to convert depth values to meters

        Returns:
            xyz1 (torch.FloatTensor): (batch_size, height * width, 4)

        Operation:
            z = d / scaling
            x = z * (u-cx) / fx
            y = z * (v-cv) / fy
        """
        shape = depth.shape
        if (
            shape[0] == self.batch_size
            and shape[1] == self.fmh
            and shape[2] == self.fmw
        ):
            x_scale = self.x_scale
            y_scale = self.y_scale
            ones = self.ones
        else:
            x_scale, y_scale, ones = self.compute_scaling_params(
                shape[0], shape[1], shape[2]
            )
        z = depth / float(depth_scaling)
        x = z * x_scale
        y = z * y_scale
        xyz1 = torch.cat((x.unsqueeze(3), y.unsqueeze(3), z.unsqueeze(3), ones), dim=3)
        return xyz1

    def transform_camera_to_world(self, xyz1, T):
        """
        Converts pointcloud from camera to world reference.

        Args:
            xyz1 (torch.FloatTensor): [(x,y,z,1)] array of N points in homogeneous coordinates
            T (torch.FloatTensor): camera-to-world transformation matrix
                                        (inverse of extrinsic matrix)

        Returns:
            (float tensor BxNx4): array of pointcloud in homogeneous coordinates

        Shape:
            Input:
                xyz1: (batch_size, 4, no_of_points)
                T: (batch_size, 4, 4)
            Output:
                (batch_size, 4, no_of_points)

        Operation: T' * R' * xyz
                   Here, T' and R' are the translation and rotation matrices.
                   And T = [R' T'] is the combined matrix provided in the function as input
                           [0  1 ]
        """
        return torch.bmm(T, xyz1)

    def pixel_to_world_mapping(self, depth_img_array, T):
        """
        Computes mapping from image pixels to 3D world (x,y,z)

        Args:
            depth_img_array (torch.FloatTensor): Depth values tensor
            T (torch.FloatTensor): camera-to-world transformation matrix (inverse of
                                        extrinsic matrix)

        Returns:
            pixel_to_world (torch.FloatTensor) : Mapping of one image pixel (i,j) in 3D world
                                                      (x,y,z)
                    array cell (i,j) : (x,y,z)
                        i,j - image pixel indices
                        x,y,z - world coordinates

        Shape:
            Input:
                depth_img_array: (N, height, width)
                T: (N, 4, 4)
            Output:
                pixel_to_world: (N, height, width, 3)
        """

        # Transformed from image coordinate system to camera coordinate system, i.e origin is
        # Camera location  # GEO:
        # shape: xyz1 (batch_size, height, width, 4)
        xyz1 = self.point_cloud(depth_img_array, self.depth_scaling)

        # shape: (batch_size, height * width, 4)
        xyz1 = torch.reshape(xyz1, (xyz1.shape[0], xyz1.shape[1] * xyz1.shape[2], 4))
        # shape: (batch_size, 4, height * width)
        xyz1_t = torch.transpose(xyz1, 1, 2)  # [B,4,HxW]

        # Transformed points from camera coordinate system to world coordinate system  # GEO:
        # shape: xyz1_w(batch_size, 4, height * width)
        xyz1_w = self.transform_camera_to_world(xyz1_t, T)

        # shape: (batch_size, height * width, 3)
        world_xyz = xyz1_w.transpose(1, 2)[:, :, :3]

        # shape: (batch_size, height, width, 3)
        pixel_to_world = torch.reshape(
            world_xyz,
            (
                (
                    depth_img_array.shape[0],
                    depth_img_array.shape[1],
                    depth_img_array.shape[2],
                    3,
                )
            ),
        )
        return pixel_to_world

    def discretize_point_cloud(self, pixels_in_world, features, camera_z_values):
        """ #GEO:
        Maps pixel in world coordinates to an (output_height, output_width) map.
        - Discretizes the (x,y) coordinates of the features to gridcellsize.
        - Remove features that lie outside the (output_height, output_width) size.
        - Computes the min_xy and size_xy, and change (x,y) coordinates setting min_xy as origin.

        Args:
            pixels_in_world (torch.FloatTensor): (x,y,z) coordinates of features in 3D world
            features (torch.FloatTensor): Image features from cnn
            camera_z_values (torch.FloatTensor): z coordinate of the camera used for deciding
                                                      after how much height to crop

        Returns:
            pixels_in_map (torch.LongTensor): World (x,y) coordinates of features discretized
                                    in gridcellsize and cropped to (output_height, output_width)
                                    and use xy_min as origin.

        Shape:
            Input:
                pixels_in_world: (batch_size, features_height, features_width, 3)
                features: (batch_size, features_height, features_width, feature_size)
                camera_z_values: (batch_size)
            Output:
                pixels_in_map: (batch_size, features_height, features_width, 2)
        """
        # Took (x,y) coordinates and discretized them in gridcellsize space
        # (and not multiplied them again with gridcellsize as they were supposed to again divided
        # by gridcellsize for computing the indices of the matrix)
        # shape: pixels_in_map (batch_size, features_height, features_width, 2)
        pixels_in_map = ((pixels_in_world[:, :, :, :2] / self.gridcellsize) +\
                # Shift origin to center of the map
                torch.tensor([self.output_width//2,self.output_height//2], device=self.device, dtype=torch.float))
        # Subtract half a grid cell if the map has an even number of cells, since the origin/center of map
        # should therefore be on the boundary of two cells (not in the middle of a cell)
        if self.output_width % 2 == 0:
            pixels_in_map[:,:,:,0] -= 0.5
        if self.output_height % 2 == 0:
            pixels_in_map[:,:,:,1] -= 0.5
        pixels_in_map = pixels_in_map.round()

        # Anything outside map boundary gets mapped to origin with an empty feature
        # mask for outside map indices
        outside_map_indices = (pixels_in_map[:, :, :, 0] >= self.output_width) +\
                              (pixels_in_map[:, :, :, 1] >= self.output_height) +\
                              (pixels_in_map[:, :, :, 0] < 0) +\
                              (pixels_in_map[:, :, :, 1] < 0)
        pixels_in_map[outside_map_indices] = 0
        # Also, set their feature values to 0
        features[outside_map_indices.unsqueeze(3).repeat(1, 1, 1, features.shape[-1])] = 0

        # shape: camera_z (batch_size, features_height, features_width)
        camera_z = (
            camera_z_values.unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, pixels_in_map.shape[1], pixels_in_map.shape[2])
        )

        # Anything above camera_z + z_clip_threshold will be ignored
        above_threshold_z_indices = pixels_in_world[:, :, :, 2] > (
            camera_z + self.z_clip_threshold
        )
        pixels_in_map[above_threshold_z_indices] = 0
        features[above_threshold_z_indices.unsqueeze(3).repeat(1, 1, 1, features.shape[-1])] = 0

        return pixels_in_map.long()


    def project_features(self, features, features_to_map, obs_per_map):
        """
        Constructs a spatial map from image features and a features to world mapping
        Element-wise Max is taken for overlapping features.

        Args:
            features (torch.FloatTensor): Image features from CNN
            feature_to_map (torch.LongTensor): Mapping from feature coords to map coords (x,y)

        Returns:
            spatial_map (torch.FloatTensor): Projected features onto 2D grid based on
                                                  mappings provided
            mask (torch.FloatTensor): Tensor of 0s and 1s where 1 tells that a non-zero
                                           feature is present at that (i,j) coordinate

        Shape:
            Input:
                features: (N, features_height, features_width, feature_size)
                feature_to_map: (N, features_height, features_width, 2)
            Output:
                spatial_map: (N/obs_per_map, output_height, output_width, feature_size)
                mask: (N/obs_per_map, output_height, output_width)

        Constraints:
            [y,x] should not exceed [output_height, output_width]
        """

        N = features.shape[0]
        feature_dim = features.shape[-1]
        flat_feat = features.reshape(-1, feature_dim)

        actual_batch_size = int(N/obs_per_map)

        feature_height = features_to_map.shape[1]
        feature_width = features_to_map.shape[2]
        batch_offset = (torch.arange(start=0, end=actual_batch_size, device=self.device).unsqueeze(1).unsqueeze(2))
        batch_offset = batch_offset.repeat(obs_per_map, feature_height, feature_width) *\
               (self.output_height * self.output_width)
        feat_index = (
            batch_offset + (self.output_width * features_to_map[:, :, :, 1] + features_to_map[:, :, :, 0]).long()
        )
        feat_index = feat_index.view(-1)  # destination ind in flat_map for each feature in flat_feat

        # https://github.com/rusty1s/pytorch_scatter
        # This will max-pool over features that are projected to the same map cell
        flat_spatial_map, _ = scatter_max(
            flat_feat,
            feat_index,
            dim=0,
            dim_size=actual_batch_size * self.output_height * self.output_width,
        )

        # shape: spatial_map (N/obs_per_map, output_height, output_width, feature_size)
        spatial_map = flat_spatial_map.reshape(
            actual_batch_size, self.output_height, self.output_width, feature_dim
        )

        # mask for non-zero feature values
        # shape: mask (N/obs_per_map, output_height, output_width)
        mask = torch.zeros(actual_batch_size, self.output_height, self.output_width, device=self.device)
        mask[spatial_map.max(dim=-1)[0] > 0] = 1
        return spatial_map, mask


    def forward(self, rgb, depth, T, obs_per_map=1):
        """Forward Function

        Args:
            rgb (torch.FloatTensor): RGB image with values in [0, 255]
            depth (torch.FloatTensor): Depth image
            T (torch.FloatTensor): camera-to-world transformation matrix
                                        (inverse of extrinsic matrix)
            obs_per_map (int): obs_per_map images are projected to the same map

        Returns:
            spatial_map (torch.FloatTensor): Projected features onto 2D grid based on
                                                  mappings computed
            mask (torch.FloatTensor): Tensor of 0s and 1s where 1 tells that a non-zero
                                           feature is present at that (i,j) coordinate
            features_to_map (torch.LongTensor): World (x,y) coordinates of features
                        discretized in gridcellsize and cropped to (output_height, output_width)
                        and use offset as origin.

        Shape:
            Input:
                rgb: (N, 3, height, width)
                depth: (N, 1, height, width)
                T: (N, 4, 4)
            Output:
                spatial_map: (N/obs_per_map, output_height, output_width, feature_size)
                mask: (N/obs_per_map, output_height, output_width)
                features_to_map: (N, features_height, features_width, 2)

         NOTE: size_xy and offset are in gridcellsize units i.e. represent grid indices
               instead of values in metres
        """

        # Compute features from CNN
        # shape: features (N, features_height, features_width, feature_size)
        features = self.cnn(rgb).permute(0, 2, 3, 1)

        # Non-zero avgpool over depth image so that (height, width) of depth img is same as CNN features
        # shape: pooled_depth_img (N, features_height, features_width)
        pooled_depth_img = self.masked_nonzero_pooling(depth).squeeze(1)

        # Feature mappings in the world coordinate system where origin is somewhere but not camera
        # # GEO:
        # shape: features_to_world (N, features_height, features_width, 3)
        features_to_world = self.pixel_to_world_mapping(pooled_depth_img, T)

        # Discretizes and computes (x,y) coordinates relative to map
        camera_z_values = T[:,2,3]
        features_to_map = self.discretize_point_cloud(features_to_world, features, camera_z_values)

        if self.add_depth_to_features:
            # Calculate normals
            normals = self.normal(self.gauss(depth)/self.depth_scaling, self.vfov)
            world_normals = torch.bmm(T[:,:3,:3], normals.flatten(start_dim=2)).reshape_as(normals)
          
            if False:
                # Save to visualize
                import cv2
                for i in range(normals.shape[0]):
                    im = normals[i].cpu().detach().numpy()
                    im2 = (im+1)*255/2.0
                    im2[im==0] = 0
                    cv2.imwrite('normals-%d.png' % i, cv2.cvtColor(im2.transpose((1,2,0)), cv2.COLOR_RGB2BGR))
            
                    im = world_normals[i].cpu().detach().numpy()
                    im2 = (im+1)*255/2.0
                    im2[im==0] = 0
                    cv2.imwrite('world_normals-%d.png' % i, cv2.cvtColor(im2.transpose((1,2,0)), cv2.COLOR_RGB2BGR))

                    im = rgb[i].cpu().detach().numpy()
                    cv2.imwrite('rgb-%d.png' % i, cv2.cvtColor(im.transpose((1,2,0)), cv2.COLOR_RGB2BGR))

                    im = depth[i].cpu().detach().numpy()
                    im = im/im.max()*255
                    cv2.imwrite('depth-%d.png' % i, cv2.cvtColor(im.transpose((1,2,0)), cv2.COLOR_GRAY2RGB))
                import pdb; pdb.set_trace()
          
            # Average the normal in non-zero areas of each feature
            pooled_normals = self.masked_nonzero_pooling(world_normals, depth).permute(0, 2, 3, 1)

            # Get the feature z-height in world coordinates relative to the camera
            relative_z = features_to_world[:,:,:,2].unsqueeze(-1) - camera_z_values.view(T.shape[0],1,1,1)

            # shape: features (N, features_height, features_width, feature_size+4)
            features = torch.cat([features, pooled_normals, relative_z], dim=3)

        # Account for zero depth (missing depth values) by zeroing those features
        features = features.where((pooled_depth_img != 0).unsqueeze(-1), torch.zeros_like(features))

        if self.add_depth_to_features:
            features = self.conv1x1(features.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Project features on spatial map based on features_to_map mappings
        # # GEO:
        # shape: spatial_map (N/obs_per_map, output_height, output_width, feature_size)
        # shape: mask (N/obs_per_map, output_height, output_width)
        spatial_map, mask = self.project_features(features, features_to_map, obs_per_map)

        return spatial_map, mask, features_to_map



def test_projection(spatial_map, features, features_to_map, obs_per_map):
    for n in range(features_to_map.shape[0]):
        for y in range(features_to_map.shape[1]):
            for x in range(features_to_map.shape[2]):
                xy = features_to_map[n,y,x]
                if xy[0] >= 0 and xy[0] < spatial_map.shape[-1] and xy[1] >= 0 and xy[1] < spatial_map.shape[-2]:
                    assert torch.all(spatial_map[n,:,xy[1],xy[0]] >= features[n,y,x])



def test_case(T, output_h, output_w, expected_ftm):

    #torch.set_printoptions(sci_mode=False)
    batch_size = 2
    feat_h = 16
    feat_w = 20

    # rgb corners are: (origin top-left)
    # green (0,1,0)       blue-green (1,1,0)
    # brown (1,1,0)       black (1,1,1)
    rgb = torch.cat([(torch.arange(0,feat_h).float()/feat_h).unsqueeze(1).unsqueeze(0).expand(-1, -1,feat_w),
                      torch.ones(1,feat_h,feat_w).float(),
                     (torch.arange(0,feat_w).float()/feat_w).unsqueeze(0).unsqueeze(0).expand(-1, feat_h,-1)], dim=0)
    rgb = rgb.unsqueeze(0).expand(batch_size, -1, -1, -1).cuda()

    # 2m away at top of image, increasing to 3.275m at bottom of image
    depth = (2.0*4000+20*torch.arange(0,256).float()).reshape(1,1,-1,1).expand(batch_size,1,-1,320).cuda()
    depth[:,:,3,:] = 0

    gridcellsize = 1 # How many metres does 1 pixel of spatial map represents
    vfov = math.radians(60) # around 72 degrees horizontal


    fp = FeatureProjection(vfov, batch_size, None, feat_h, feat_w, output_h, output_w, gridcellsize, 10, False)
    features = rgb.permute(0, 2, 3, 1)
    pooled_depth_img = fp.masked_nonzero_pooling(depth).squeeze(1)
    features_to_world = fp.pixel_to_world_mapping(pooled_depth_img, T)
    camera_z_values = T[:,2,3]
    features_to_map = fp.discretize_point_cloud(features_to_world, features, camera_z_values)
    assert torch.all(torch.eq(features_to_map, expected_ftm))

    proj, mask = fp.project_features(features, features_to_map, obs_per_map=1)
    spatial_map = proj.permute(0, 3, 1, 2)
    test_projection(spatial_map, features, features_to_map, 1)


if __name__ == "__main__":

    # Image origin is top-left, camera looks down z-axis
    T = torch.FloatTensor([[[ 1, 0, 0, 0], # Camera looking down y-axis, z-up, x-right
                            [ 0, 0, 1, 0],
                            [ 0,-1, 0, 0],
                            [ 0, 0, 0, 1]],
                           [[ 0, 0, 1, 0], # Camera looking down x-axis, z-up, y-right
                            [-1, 0, 0, 0],
                            [ 0,-1, 0, 0],
                            [ 0, 0, 0, 1]]]).cuda()

    # Expected mapping of features from image to semantic map for 7x7 map
    ftm_x = torch.tensor([[[2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4],
                           [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4],
                           [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5],
                           [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5],
                           [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5],
                           [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5],
                           [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5],
                           [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5],
                           [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5],
                           [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5],
                           [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
                           [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
                           [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
                           [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
                           [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
                           [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]],

                          [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]]],
                          device='cuda:0')
    ftm_y = torch.tensor([[[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]],

                          [[4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2],
                           [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2],
                           [5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1],
                           [5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1],
                           [5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1],
                           [5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1],
                           [5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1],
                           [5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1],
                           [5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1],
                           [5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1],
                           [5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1],
                           [5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1],
                           [5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1],
                           [5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1],
                           [5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1],
                           [5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]]],
                          device='cuda:0')
    ftm = torch.cat([ftm_x.unsqueeze(-1), ftm_y.unsqueeze(-1)], dim=-1)
    test_case(T, 7, 7, ftm)

    # Expected mapping of features from image to semantic map for 8x8 map
    ftm_x = torch.tensor([[[2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5],
                          [2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5],
                          [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
                          [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
                          [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
                          [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
                          [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
                          [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
                          [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
                          [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
                          [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
                          [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6],
                          [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6],
                          [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6],
                          [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6],
                          [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6]],

                         [[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                          [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                          [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]]],
                          device='cuda:0')
    ftm_y = torch.tensor([[[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                          [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                          [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                          [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]],

                         [[5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2],
                          [5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2],
                          [5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2],
                          [5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2],
                          [5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2],
                          [5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2],
                          [5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2],
                          [5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2],
                          [5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2],
                          [5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2],
                          [5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2],
                          [6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1],
                          [6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1],
                          [6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1],
                          [6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1],
                          [6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1]]],
                          device='cuda:0')
    ftm = torch.cat([ftm_x.unsqueeze(-1), ftm_y.unsqueeze(-1)], dim=-1)
    test_case(T, 8, 8, ftm)
