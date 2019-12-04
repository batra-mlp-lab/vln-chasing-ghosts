import copy
import torch
import torch.nn.functional as F
from torch import optim
import cv2
import numpy as np
import math

import utils
import cfg
from simulator import PanoSimulator
from floorplan import Floorplan
from dataloader import DataLoader
from modules.optim import AdamW
from visdom_wrapper import VisdomVisualize

# rgba
autumn = np.array([[1.        , 0.        , 0.        , 1.        ],
                   [1.        , 0.16470588, 0.        , 1.        ],
                   [1.        , 0.33333333, 0.        , 1.        ],
                   [1.        , 0.50196078, 0.        , 1.        ],
                   [1.        , 0.66666667, 0.        , 1.        ],
                   [1.        , 0.83529412, 0.        , 1.        ],
                   [1.        , 1.        , 0.        , 1.        ]])
autumn = np.flip(autumn[:,:3], axis=-1)
autumn = np.flip(autumn, axis=0) # yellow to red looks better than red to yellow

winter = np.array([[0.        , 0.        , 1.        , 1.        ],
                   [0.        , 0.16470588, 0.91764706, 1.        ],
                   [0.        , 0.33333333, 0.83333333, 1.        ],
                   [0.        , 0.50196078, 0.74901961, 1.        ],
                   [0.        , 0.66666667, 0.66666667, 1.        ],
                   [0.        , 0.83529412, 0.58235294, 1.        ],
                   [0.        , 1.        , 0.5       , 1.        ]])
winter = np.flip(winter[:,:3], axis=-1)


class Trainer(object):
    """ Trainer class containing boilerplate code for dataloading, Simulator init and overloading
        eval params """

    def __init__(self, args, filepath=None, load_sim=True):
        self.args = copy.deepcopy(args)
        if filepath:
            # Copy args from the loaded config to self.args. This ensures old models can still
            # be loaded even after new args are added to the config
            self.loader = torch.load(filepath)
            print("Loading from checkpoint: %s" % filepath)
            for arg,value in vars(self.loader["args"]).items():
                setattr(self.args, arg, value)

            # eval parameters override
            self.args.filepath = filepath
            self.args.eval = args.eval
            self.args.val_epoch = args.val_epoch
            self.args.batch_size = args.batch_size
            self.args.viz_eval = args.viz_eval
            self.args.viz_gif = args.viz_gif
            self.args.viz_iterations = args.viz_iterations
            self.args.viz_folder = args.viz_folder
            self.args.visdom = args.visdom
            self.args.visdom_env = args.visdom_env
            self.args.visdom_server = args.visdom_server
            self.args.visdom_port = args.visdom_port
            self.args.preloading = args.preloading
            self.args.debug_scale = args.debug_scale
            self.args.debug_mode = args.debug_mode
            self.args.start_epoch = args.start_epoch

        if self.args.visdom:
            visdom_env = self.args.exp_name if self.args.visdom_env == "" else self.args.visdom_env
            self.visdom = VisdomVisualize(
                env_name=visdom_env, win_prefix=self.args.exp_name + "_",
                server=self.args.visdom_server, port=self.args.visdom_port
            )
            self.visdom.text(cfg.print_args(self.args, verbose=False).replace("\n", "<br>"))
        else:
            self.visdom = None

        # Set random seed before anything
        seed = self.args.random_seed
        if self.args.start_epoch != 1 and self.args.eval==False:
            seed = self.args.start_epoch + seed
        utils.set_random_seed(seed)

        if load_sim:
            self.sim = PanoSimulator(self.args)

        # Load data (trajectories, scans, graphs, vocab)
        if not self.args.eval:
            splits = ['train', 'val_seen', 'val_unseen'] if self.args.train_split == 'trainval' else ['train']
            self.traindata = DataLoader(self.args, splits=splits)
        self.valseendata = DataLoader(self.args, splits=['val_seen'])
        self.valunseendata = DataLoader(self.args, splits=['val_unseen'])

        self.all_scans = self.valseendata.scans | self.valunseendata.scans
        if not self.args.eval:
            self.all_scans |= self.traindata.scans
        self.floorplan = Floorplan(self.args.debug_scale/self.args.gridcellsize, list(self.all_scans))


    def optimizer(self, model_params):
        if self.args.optimizer == 'AdamW':
            opt = AdamW(model_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            opt = getattr(optim, self.args.optimizer)
            opt = opt(model_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return opt


    def floor_maps(self, states):
        """ Get a list of floorplan maps centered at the provided states
            Input:
                states: simulator states (at the beginning of the episode!)
        """
        s = self.args.debug_scale
        ims = self.floorplan.rgb(states, (s*self.args.map_range_x,s*self.args.map_range_y))
        base_maps = []
        for n,im in enumerate(ims):
            im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR).astype(np.float64)/255.0
            base_maps.append(im)
        return base_maps


    def overlay_mask(self, base_maps, mask):
        """ Overlay spatial map mask onto existing base maps
            Input:
                base_maps: list of numpy BGR floating point images
                mask: spatial map mask pytorch tensor
        """
        mask = (torch.flip(mask, [2])).squeeze(1).cpu().detach().numpy()
        new_maps = []
        for n,im in enumerate(base_maps):
            nm = cv2.resize(mask[n], (im.shape[1], im.shape[0]), interpolation = cv2.INTER_NEAREST)
            new_map = (0.4*nm[...,np.newaxis]*im + 0.4*im + 0.2*nm[...,np.newaxis])
            new_maps.append(new_map)
        return new_maps


    def overlay_belief(self, base_maps, belief, show_argmax=True, colormap=autumn):
        """ Overlay belief predictions onto existing base maps
            Input:
                base_maps: list of numpy BGR floating point images
                belief: pytorch tensor (N, K, heading_states, Y, X)
        """
        belief = (torch.flip(belief, [-2])).cpu().detach().numpy()
        new_maps = []
        for n,im in enumerate(base_maps):
            for k in range(belief.shape[1]):
                bel_ = belief[n,k].sum(axis=0)
                bel = cv2.resize(bel_, (im.shape[1], im.shape[0]), interpolation = cv2.INTER_LINEAR)
                bel = 0.3*bel/bel.max() # apply some consistent scaling to make it more visible
                im = bel[...,np.newaxis]*colormap[k] + (1-bel[...,np.newaxis])*im
            for k in range(belief.shape[1]):
                bel_ = belief[n,k].sum(axis=0)
                bel = cv2.resize(bel_, (im.shape[1], im.shape[0]), interpolation = cv2.INTER_LINEAR)
                if show_argmax:
                    radius = int(max(1,im.shape[0]/256))
                    line_thickness = int(max(1,im.shape[0]/1024))
                    bel_pos = np.unravel_index(cv2.GaussianBlur(bel, (5,5), 1).argmax(), bel.shape)
                    if bel_.shape[0] > 1: # has heading
                        pos = np.unravel_index(bel_.argmax(), bel_.shape)
                        heading_probs = belief[n,k,:,pos[0],pos[1]]
                        heading = heading_probs.argmax()
                        offset = max(2*radius,radius+2)
                        # After flipping image, x is right but y is up
                        if heading==0:
                            offset = (bel_pos[1]+offset, bel_pos[0])
                        elif heading==1:
                            offset = (bel_pos[1], bel_pos[0]-offset)
                        elif heading==2:
                            offset = (bel_pos[1]-offset, bel_pos[0])
                        else:
                            offset = (bel_pos[1], bel_pos[0]+offset)
                        cv2.line(im, (bel_pos[1], bel_pos[0]), offset, (1,1,1), thickness=line_thickness) # line to show heading
                    cv2.circle(im,(bel_pos[1],bel_pos[0]), radius+line_thickness, (1,1,1), -1) # center (agent start)
                    cv2.circle(im,(bel_pos[1],bel_pos[0]), radius, colormap[k], -1) # center (agent start)
            new_maps.append(im)
        return new_maps


    def overlay_local_graph(self, base_maps, features, action_idx=None):
        """ Overlay observed local graphs onto existing base maps
            Input:
                base_maps: list of numpy BGR floating point images
                features: graph nodes from simulator
                action_idx: optional next step
        """
        new_maps = []
        radius = int(max(1,base_maps[0].shape[0]/256))
        map_range_m_x = self.args.map_range_x * self.args.gridcellsize
        map_range_m_y = self.args.map_range_y * self.args.gridcellsize
        for n,im in enumerate(base_maps):
            new_maps.append(im.copy())
        last_n = -1
        idx = 0
        for node in features:
            n = int(node[0])
            if n > last_n:
                idx = 0
            else:
                idx += 1
            if action_idx is not None and action_idx[n] == idx:
                color = (0,0,1) # red dot for next step
            elif node[-1] == 1: # visited
                color = (0,1,0) # green dot
            else:
                color = (1,0,0) # blue dots for graph
            xy = node[1:3].cpu().numpy()
            im = new_maps[n]
            x_pos = int(round((0.5 + xy[0]/map_range_m_x)*im.shape[1]))
            # After flipping image, x is right but y is up
            y_pos = int(round((0.5 - xy[1]/map_range_m_y)*im.shape[0]))
            cv2.circle(im, (x_pos,y_pos), radius, color, -1)
            last_n = n
        return new_maps


    def overlay_goal(self, base_maps, goal_coords):
        new_maps = []
        radius = int(max(3,base_maps[0].shape[0]/128))
        map_range_m_x = self.args.map_range_x * self.args.gridcellsize
        map_range_m_y = self.args.map_range_y * self.args.gridcellsize
        for n,im in enumerate(base_maps):
            im = im.copy()
            x_pos = int(round(((0.5 + goal_coords[n,0]/map_range_m_x)*im.shape[1]).item()))
            # After flipping image, x is right but y is up
            y_pos = int(round(((0.5 - goal_coords[n,1]/map_range_m_y)*im.shape[0]).item()))
            cv2.circle(im, (x_pos,y_pos), radius, (0,0.65,1), -1)
            cv2.circle(im, (x_pos,y_pos), radius-2, (0,0.85,1), -1)
            new_maps.append(im)
        return new_maps


    def feature_sources(self, rgb, ftm, map_x, map_y):
        """ Return a list of images showing features that map to these map coords """
        valid_features = ((ftm == torch.tensor([map_x, map_y]).to(ftm.device)).sum(dim=-1)==2)
        valid_images = valid_features.reshape(valid_features.shape[0],-1).sum(dim=1) != 0
        valid_pixels = F.interpolate(valid_features.float().unsqueeze(1), size=rgb.shape[-2:], mode='nearest')
        ims = rgb.clone()
        ims[~valid_pixels.byte().expand(-1,3,-1,-1)] *= 0.5
        ims = ims[valid_images]
        ims = (torch.flip(ims, [1])).permute(0,2,3,1).cpu().detach().numpy()
        return ims


    def visual_debug(self, t, rgb, mask, debug_map, input_belief, belief, target_belief):
        s = self.args.debug_scale
        n = 0
        if t == 0:
            im = self.floorplan.rgb(self.sim.getState(), (s*self.args.map_range_x,s*self.args.map_range_y))[n]
            im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
            cv2.circle(im,(im.shape[1]/2, im.shape[0]/2), 1, (0,0,255), -1) # center (agent start)
            cv2.imshow('Floorplan', im)

        rgb = rgb.flip((1)).cpu().detach().numpy()[n]
        cv2.imshow('RGB', rgb.transpose((1,2,0))/255.0)

        mask = (torch.flip(mask, [2])).cpu().detach().numpy()[n]

        # Flips are to flip the y axis
        belief = (torch.flip(belief, [2])).cpu().detach().numpy()[n]
        belief /= belief.max()
        input_belief = (torch.flip(input_belief, [2])).cpu().detach().numpy()[n]
        input_belief /= input_belief.max()
        target_belief = (torch.flip(target_belief, [2])).cpu().detach().numpy()[n]
        target_belief /= target_belief.max()
        vis_belief = np.stack([input_belief,belief, target_belief], axis=3)

        for i in range(vis_belief.shape[0]):
            bel = 0.8*vis_belief[i] + 0.2*mask.transpose((1,2,0))
            cv2.imshow('Belief (blue-input, green-predict, red-target) %d' % i, cv2.resize(bel, (0,0), fx=s, fy=s))

        debug_map = (torch.flip(debug_map, [2])).cpu().detach().numpy()[n]
        merge_map = 0.8*debug_map + 0.2*mask
        cv2.imshow('Debug Map', cv2.resize(merge_map.transpose((1,2,0)), (0,0), fx=s, fy=s))  # (C,H,W) to (H,W,C)
        cv2.waitKey(0)
