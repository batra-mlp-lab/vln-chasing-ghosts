
import argparse
import math
import text_utils

parser = argparse.ArgumentParser()

# directories
parser.add_argument("-data_dir", type=str, default="tasks/R2R/data", help="Path to training data")
parser.add_argument("-result_dir", type=str, default="tracker/results", help="Path to store results")
parser.add_argument("-snapshot_dir", type=str, default="tracker/snapshots", help="Path to store model snapshots")
parser.add_argument("-viz_folder", type=str, default="tracker/viz", help="Folder to save visualizations to. If not specified, defaults to viz/exp_name")
parser.add_argument("-train_vocab", type=str, default="tracker/data/train_vocab_v1.txt", help="Filename for training vocab")
parser.add_argument("-trainval_vocab", type=str, default="tracker/data/trainval_vocab_v1.txt", help="Filename for training + validation vocab")

# agent / simulator parameters
parser.add_argument("-device", type=str, default="cuda", help="The device on which PyTorch Tensors will be allocated")
parser.add_argument("-elevation_degrees", type=float, default=-30, help="Elevation of agent in Degrees for first pano sweep")
parser.add_argument("-elevation_chg_degrees", type=float, default=50, help="Elevation increase in Degrees on next pano sweep")
parser.add_argument("-num_pano_sweeps", type=int, default=1, help="The number of sweeps of num_pano_views")
parser.add_argument("-num_pano_views", type=int, default=12, help="The number of views comprising a 360 degree pano")
parser.add_argument("-vfov_degrees", type=float, default=60, help="Vertical field of view in radians")
parser.add_argument('-image_width', type=int, default=320, help='Width of image from simulator')
parser.add_argument('-image_height', type=int, default=256, help='Height of image from simulator')
parser.add_argument('-preloading', action='store_true', default=False, help='Whether preload images for speedup or not')
parser.add_argument('-cnn_feature_map_height', type=int, default=16, help='Height of feature map from CNN')  # Must match the image size / CNN output
parser.add_argument('-cnn_feature_map_width', type=int, default=20, help='Width of feature map from CNN')
parser.add_argument('-finetune_cnn', dest='finetune_cnn', action='store_true', default=False, help='Finetune the CNN')
parser.add_argument('-z_clip_threshold', type=float, default=5.0, help='Consider pixels only lying within the z threshold above the camera (in metres)')
parser.add_argument('-random_seed', type=int, default=0, help='Random seed impacting minibatch sampling and agent motion')
parser.add_argument('-supervision_prob', type=float, default=0.5, help='Probability with which the agent follows the training path. Negative values for a declining schedule.')
parser.add_argument('-timesteps', type=int, default=6, help='Max timesteps for which agent is allowed to run')
parser.add_argument('-max_graph_degree', type=int, default=13, help='Maximum degree of all nodes in the navigation graphs')
parser.add_argument('-depth_scaling', type=float, default=4000.0, help='Matterport depth scaling 0.25 mm per value (divide by 4000 to get meters)')

# mapper parameters
parser.add_argument('-kernel_size', type=int, default=3, help='Kernel size of convolution in spatial map update')
parser.add_argument("-gridcellsize", type=float, default=0.5, help="How many meters does one cell of the spatial map represent")
parser.add_argument('-map_depth', type=int, default=128, help='No. of features in spatial map')
parser.add_argument('-map_range_y', type=int, default=96, help='Y-size of spatial map, should be an even value')
parser.add_argument('-map_range_x', type=int, default=96, help='X-size of spatial map, should be an even value')
parser.add_argument("-map_feature_size", type=int, default=768, help="Size of CNN features used in map.")
parser.add_argument('-add_depth_to_map_features', action='store_true', default=False, help='Add normals etc to map features.')
parser.add_argument('-mapper_dropout_ratio', type=float, default=0.5, help='Dropout in the mapper')

# encoder parameters
parser.add_argument("-word_embedding_size", type=int, default=300, help="word embedding dimension in instruction encoder")
parser.add_argument("-enc_hidden_size", type=int, default=256, help="LSTM language encoder hidden state size (in each direction for bidirectional)")
parser.add_argument("-enc_lstm_layers", type=int, default=1, help="Layers in the language encoder LSTM")
parser.add_argument("-bidirectional", action="store_true", default=True, help="Flag to make instruction encoder bidirectional")
parser.add_argument("-enc_dropout_ratio", type=float, default=0.0, help="Dropout in language encoder LSTM")

# filter parameters
parser.add_argument('-heading_states', type=int, default=4, help='Number of heading states (typically 1 for no heading, or 4).')
parser.add_argument('-belief_downsample_factor', type=int, default=2, help='Divide map_range by this to get belief dimensions')
parser.add_argument('-motion_kernel_size', type=int, default=7, help='Motion kernel size')
parser.add_argument('-motion_hidden_size', type=int, default=96, help='Num channels in motion model')
parser.add_argument('-motion_map_sparse', action='store_true', default=False, help='Use sparse convolution in motion model')
parser.add_argument('-motion_map_bias', action='store_true', default=False, help='Use convolution plus bias in motion model')
parser.add_argument("-motion_dropout_ratio", type=float, default=0.5, help="Dropout in the motion model")
parser.add_argument('-motion_occupancy_map', action='store_true', default=False, help='Motion model uses a single channel occupancy map only')
parser.add_argument('-min_obs_likelihood', type=float, default=0.004, help="Minimum observation likelihood")
parser.add_argument('-truncate_after', type=int, default=3, help='Truncate gradients after this many timesteps')
parser.add_argument('-norm_layer', type=str, default='instance_norm', help='Which type of normalization to use in LingUNet. "instance_norm"/"batch_norm"')

# decoder/policy parameters
parser.add_argument("-dec_hidden_size", type=int, default=256, help="Decoder hidden state size")
parser.add_argument("-att_hidden_size", type=int, default=256, help="Number of attention hidden units")
parser.add_argument("-action_embedding_size", type=int, default=64, help="action/positional embedding dimension")
parser.add_argument("-policy_hidden_size", type=int, default=128, help="Size of policy hidden state")
parser.add_argument("-policy_loss_lambda", type=float, default=0.1, help="Weight on policy loss")
parser.add_argument('-policy_gt_belief', action='store_true', default=False, help='Policy uses ground-truth belief maps (to establish upper bound)')

# Handcoded Baseline parameters
parser.add_argument("-handcoded_radius", type=float, default=7.5, help="Radius (in metres) for handcoded baseline")
parser.add_argument("-blur_kernel_x", type=int, default=45, help="Kernel size (in pixel units) to apply gaussian blur along x-axis (width). Should be odd integer")
parser.add_argument("-blur_kernel_y", type=int, default=45, help="Kernel size (in pixel units) to apply gaussian blur along y-axis (height). Should be odd integer")

# training parameters
parser.add_argument('-train_split', type=str, default='train', help='train or trainval')
parser.add_argument("-batch_size", type=int, default=5, help="Training Batch size")
parser.add_argument("-max_iterations", type=int, default=15000, help="No of training iterations")
parser.add_argument("-start_epoch", type=int, default=1, help="Epoch to resume traing from")
parser.add_argument("-max_input_length", type=int, default=80, help="Max words in instruction")
parser.add_argument("-max_steps", type=int, default=7, help="Max viewpoint steps in a path to goal, including the start and goal viewpoints")
parser.add_argument("-optimizer", type=str, default='Adam', help='PyTorch optimizers (e.g. Adam, SGD) plus AdamW')
parser.add_argument("-learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("-weight_decay", type=float, default=1e-7, help="Weight decay")
parser.add_argument("-action_ignore_index", type=int, default=-1, help="Ignore value for cross entropy loss")
parser.add_argument("-goal_heatmap_sigma", type=float, default=2.0, help="Standard deviation (in meters) for determining training targets for goal map")
parser.add_argument("-path_heatmap_sigma", type=float, default=2.0, help="Standard deviation (in meters) for determining training targets for path map")
parser.add_argument("-filter_heatmap_sigma", type=float, default=0.5, help="Standard deviation (in meters) for filter training target belief")
parser.add_argument("-filter_input_sigma", type=float, default=0.5, help="Standard deviation (in meters) for filter training input belief")
parser.add_argument('-multi_maps', action='store_true', default=True, help='Train for each partially observable map or not')
parser.add_argument("-teacher_force_motion_model", action='store_true', default=False, help="Prevent bad self-feedback on motion model")

# evaluation parameters
parser.add_argument("-eval", action="store_true", default=False, help="Either train the model or evaluate model on val_epoch")
parser.add_argument("-val_epoch", type=int, default=3000, help="Which epoch to eval in the end")

# logging parameters
parser.add_argument("-exp_name", type=str, default="Experiment", help="Experiment name (for logging, plots)")
parser.add_argument("-visdom", action="store_true", default=False, help="Use visdom or not")
parser.add_argument("-visdom_env", type=str, default="", help="Visdom Environment. Defaults to exp_name. ")
parser.add_argument("-visdom_server", type=str, default="http://localhost", help="Hostname of Visdom server")
parser.add_argument("-visdom_port", type=int, default=8097, help="Port on which Visdom server is running")
parser.add_argument("-log_every", type=int, default=10, help="When to print loss")
parser.add_argument("-validate_every", type=int, default=500, help="When to calculate validation loss")
parser.add_argument("-validation_iterations", type=int, default=100, help="How many batches when validating")
parser.add_argument("-checkpoint_every", type=int, default=500, help="When to plot and checkpoint model weights")
parser.add_argument("-viz_eval", action="store_true", default=False, help="Whether to save few visualizations while evaluation or not.")
parser.add_argument("-viz_gif", action="store_true", default=False, help="Save GIFs instead of images.")
parser.add_argument("-viz_iterations", type=int, default=10, help="How many iterations to save visualizations for.")
parser.add_argument("-debug_mode", action="store_true", default=False, help="Show visual output.")
parser.add_argument("-debug_scale", type=int, default=8, help="Size of visual output.")


def print_args(args, verbose=True):
    s = ""
    args_dict = vars(args)
    for key, value in sorted(args_dict.items()):
        if args.eval is False and verbose:
            print(key, value)
        s+= "%s: %s\n" % (key, value)
    return s


def parse_args():
    args = parser.parse_args()
    args.enc_padding_idx = text_utils.BASE_VOCAB.index("<PAD>")
    args.cache_size = 2*args.batch_size
    args.vfov = math.radians(args.vfov_degrees)
    args.hfov = 2*math.atan(float(args.image_width)/args.image_height*math.tan(args.vfov/2.0))
    args.elevation = math.radians(args.elevation_degrees)
    args.elevation_chg = math.radians(args.elevation_chg_degrees)
    if args.viz_gif:
        args.debug_scale = 16
    return args
