import torch
import torch.nn.functional as F

import os
import time
import random
import cv2

import utils
import viz_utils
import cfg
from trainer import Trainer
from evaluator import Evaluator
from simulator import PanoSimulatorWithGraph
from modules.mapper import Mapper
from modules.navigators import Filter
from modules.policy import ReactivePolicy
from modules.loss import LogBeliefLoss
from train_policy import belief_at_nodes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class FilterTrainer(Trainer):
    """ Train a filter """

    def __init__(self, args, filepath=None):
        super(FilterTrainer, self).__init__(args, filepath)

        # load models
        self.mapper = Mapper(self.args).to(self.args.device)
        self.model = Filter(self.args).to(self.args.device)

        self.map_opt = self.optimizer(self.mapper.parameters())
        self.model_opt = self.optimizer(self.model.parameters())

        if filepath:
            loader = torch.load(filepath)
            self.mapper.load_state_dict(loader["mapper"])
            self.model.load_state_dict(loader["filter"])
            self.map_opt.load_state_dict(loader["mapper_optimizer"])
            self.model_opt.load_state_dict(loader["filter_optimizer"])
            print("Loaded Mapper and Filter from: %s" % filepath)
        elif args:
            self.model.init_weights()
        self.criterion = LogBeliefLoss()


    def validate(self, split):
        """
        split: "val_seen" or "val_unseen"
        """
        with torch.no_grad():

            if(split=="val_seen"):
                val_dataloader = self.valseendata
            elif(split=="val_unseen"):
                val_dataloader = self.valunseendata

            total_loss = torch.tensor(0.0)
            normalizer = 0

            for it in range(self.args.validation_iterations):

                # Load minibatch and simulator
                seq, seq_mask, seq_lens, batch = val_dataloader.get_batch()
                self.sim.newEpisode(batch)

                # Initialize the mapper
                xyzhe = self.sim.getXYZHE()
                spatial_map,mask = self.mapper.init_map(xyzhe)

                # Note mm is being validated on the GT path. This is not the path taken by Mapper.
                path_xyzhe, path_len = val_dataloader.path_xyzhe()

                for t in range(self.args.timesteps):
                    rgb,depth,states = self.sim.getPanos()
                    spatial_map,mask,ftm = self.mapper(rgb,depth,states,spatial_map,mask)

                    state = None
                    steps = self.args.max_steps-1

                    belief = self.mapper.belief_map(path_xyzhe[0], self.args.filter_input_sigma).log()
                    for k in range(steps):
                        input_belief = belief
                        # Train a filter
                        new_belief,obs_likelihood,state,_,_,_ = self.model(k, seq, seq_mask, seq_lens, input_belief, spatial_map, state)
                        belief = new_belief + obs_likelihood
                        # Renormalize
                        belief = belief - belief.reshape(belief.shape[0], -1).logsumexp(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)

                        # Determine target and loss
                        target_heatmap = self.mapper.belief_map(path_xyzhe[k+1], self.args.filter_heatmap_sigma)
                        # To make loss independent of heading_states, we sum over all heading states (logsumexp) for validation
                        total_loss += self.criterion(belief.logsumexp(dim=1).unsqueeze(1), target_heatmap.sum(dim=1).unsqueeze(1))
                        normalizer += self.args.batch_size

                    # Take action in the sim
                    self.sim.takePseudoSupervisedAction(val_dataloader.get_supervision)

        return total_loss/normalizer

    def logging_loop(self, it):
        """ Logging and checkpointing stuff """

        if it % self.args.validate_every == 0:
            self.mapper.eval()
            self.model.eval()
            loss_val_seen = self.validate("val_seen")
            loss_val_unseen = self.validate("val_unseen")

            self.prev_time, time_taken = utils.time_it(self.prev_time)

            print("Iteration: %d Loss: %f Val Seen Loss: %f Val Unseen Loss: %f Time: %0.2f secs"
                    %(it, self.loss.item(), loss_val_seen.item(), loss_val_unseen.item(), time_taken))

            if self.visdom:
                # visdom: X, Y, key, line_name, x_label, y_label, fig_title
                self.visdom.line(it, self.loss.item(), "train_loss", "Train Loss", "Iterations", "Loss", title=" Train Phase")
                self.visdom.line(it, loss_val_seen.item(), "val_loss", "Val Seen Loss", "Iterations", "Loss", title="Val Phase")
                self.visdom.line(it, loss_val_unseen.item(), "val_loss", "Val Unseen Loss", "Iterations", "Loss", title="Val Phase")
            self.mapper.train()
            self.model.train()

        elif it % self.args.log_every == 0:
            self.prev_time, time_taken = utils.time_it(self.prev_time)
            print("Iteration: %d Loss: %f Time: %0.2f secs" % (it, self.loss.item(), time_taken))
            if self.visdom:
                self.visdom.line(it, self.loss.item(), "train_loss", "Train Loss", "Iterations", "Loss", title="Train Phase")

        if it % self.args.checkpoint_every == 0:
            saver = {
                        "mapper": self.mapper.state_dict(),
                        "filter": self.model.state_dict(),
                        "mapper_optimizer": self.map_opt.state_dict(),
                        "filter_optimizer": self.model_opt.state_dict(),
                        "epoch": it,
                        "args": self.args,
                    }
            dir = "%s/%s" % (self.args.snapshot_dir, self.args.exp_name)
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(
                saver, "%s/%s_%d" % (dir, self.args.exp_name, it)
            )
            if self.visdom:
                self.visdom.save()


    def map_to_image(self):
        """ Show where map features are found in the RGB images """
        traindata = self.traindata

        # Load minibatch and simulator
        seq, seq_mask, seq_lens, batch= traindata.get_batch(False)
        self.sim.newEpisode(batch)

        # Initialize the mapper
        xyzhe = self.sim.getXYZHE()
        spatial_map,mask = self.mapper.init_map(xyzhe)

        for t in range(self.args.timesteps):
            rgb,depth,states = self.sim.getPanos()

            # ims = (torch.flip(rgb, [1])).permute(0,2,3,1).cpu().detach().numpy()
            # for n in range(ims.shape[0]):
            #     cv2.imwrite('im %d.png' % n, ims[n])
            spatial_map,mask,ftm = self.mapper(rgb,depth,states,spatial_map,mask)

            im_ix = torch.arange(0, ftm.shape[0], step=self.args.batch_size).to(ftm.device)
            ims = self.feature_sources(rgb[im_ix], ftm[im_ix], 56, 48)
            for n in range(ims.shape[0]):
                cv2.imwrite('im %d-%d.png' % (t,n), ims[n])

            # Take action in the sim
            self.sim.takePseudoSupervisedAction(traindata.get_supervision)


    def train(self):
        """
        Supervised training of the filter
        """

        torch.autograd.set_detect_anomaly(True)
        self.prev_time = time.time()

        # Set models to train phase
        self.mapper.train()
        self.model.train()

        for it in range(self.args.start_epoch, self.args.max_iterations + 1):

            self.map_opt.zero_grad()
            self.model_opt.zero_grad()

            traindata = self.traindata
            # Load minibatch and simulator
            seq, seq_mask, seq_lens, batch= traindata.get_batch()
            self.sim.newEpisode(batch)
            if self.args.debug_mode:
                debug_path = traindata.get_path()

            # Initialize the mapper
            xyzhe = self.sim.getXYZHE()

            spatial_map,mask = self.mapper.init_map(xyzhe)

            # Note model is being trained on the GT path. This is not the path taken by Mapper.
            path_xyzhe, path_len = traindata.path_xyzhe()

            loss = 0
            normalizer = 0

            for t in range(self.args.timesteps):
                rgb,depth,states = self.sim.getPanos()

                spatial_map,mask,ftm = self.mapper(rgb,depth,states,spatial_map,mask)
                del states
                del depth
                del ftm
                if not self.args.debug_mode:
                    del rgb

                state = None
                steps = self.args.max_steps-1

                belief = self.mapper.belief_map(path_xyzhe[0], self.args.filter_input_sigma).log()
                for k in range(steps):
                    input_belief = belief
                    # Train a filter
                    new_belief,obs_likelihood,state,_,_,_ = self.model(k, seq, seq_mask, seq_lens, input_belief, spatial_map, state)
                    belief = new_belief + obs_likelihood
                    # Renormalize
                    belief = belief - belief.reshape(belief.shape[0], -1).logsumexp(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)

                    # Determine target and loss
                    target_heatmap = self.mapper.belief_map(path_xyzhe[k+1], self.args.filter_heatmap_sigma)
                    loss += self.criterion(belief, target_heatmap)
                    normalizer += self.args.batch_size

                    if self.args.debug_mode:
                        input_belief = F.interpolate(input_belief.exp(), scale_factor=self.args.belief_downsample_factor,
                                              mode='bilinear', align_corners=False).sum(dim=1).unsqueeze(1)
                        belief_up = F.interpolate(belief.exp(), scale_factor=self.args.belief_downsample_factor,
                                              mode='bilinear', align_corners=False).sum(dim=1).unsqueeze(1)
                        target_heatmap_up = F.interpolate(target_heatmap, scale_factor=self.args.belief_downsample_factor,
                                              mode='bilinear', align_corners=False).sum(dim=1).unsqueeze(1)
                        debug_map = self.mapper.debug_maps(debug_path)

                        self.visual_debug(t, rgb, mask, debug_map, input_belief, belief_up, target_heatmap_up)

                trunc = self.args.truncate_after
                if (t%trunc)==(trunc-1) or t+1 == self.args.timesteps:
                    self.loss = loss/normalizer
                    self.loss.backward()
                    self.loss.detach_()
                    loss = 0
                    normalizer = 0
                    spatial_map.detach_()
                    if state is not None:
                        state = self.model.detach_state(state)
                        state = None # Recalc to get gradients from later in the decoding - #TODO keep this part of the graph without re-running the forward part

                # Take action in the sim
                self.sim.takePseudoSupervisedAction(traindata.get_supervision)

            del spatial_map
            del mask

            self.map_opt.step()
            self.model_opt.step()
            self.logging_loop(it)


class FilterEvaluator(Evaluator):

    def __init__(self, args, filepath):
        super(FilterEvaluator, self).__init__(args, filepath)

    def init_models(self):
        self.mapper = Mapper(self.args).to(self.args.device)
        self.model = Filter(self.args).to(self.args.device)

    def load_model_weights(self):
        self.mapper.load_state_dict(self.loader["mapper"])
        self.model.load_state_dict(self.loader["filter"])
        print("Loaded Mapper and Filter from: %s" % filepath)

    def set_models_eval(self):
        self.mapper.eval()
        self.model.eval()

    def viz_attention_weights(self, act_att_weights, obs_att_weights, seqs, seq_lens, split, it, save_folder="viz"):
        # act_att_weights: (N, T, self.args.max_steps-1, torch.max(seq_lens))
        # seqs: (N, max_seq_len)
        # seq_len (N,)
        batch_size = act_att_weights.shape[0]
        timesteps = act_att_weights.shape[1]
        for n in range(batch_size):
            instruction = self.dataloader.tokenizer.decode_sentence(seqs[n]).split()
            for t in range(timesteps):
                act_att = act_att_weights[n][t].cpu().numpy()
                obs_att = obs_att_weights[n][t].cpu().numpy()
                valid_act_att = act_att[:, :seq_lens[n]].transpose()
                valid_obs_att = obs_att[:, :seq_lens[n]].transpose()
                fig = viz_utils.plot_att_graph(valid_act_att, valid_obs_att, instruction, seq_lens[n].item())
                if t == 0: # Currently doesn't use the map, so can just save once
                    fig.savefig("%s/attention-%s-it%d-n%d-t%d.png" % (save_folder, split, it, n, t))
                plt.close('all')
        print("Saved Attention viz: %s/attention-%s-it%d-n-t.png" % (save_folder, split, it))

    def eval_viz(self, goal_pred, path_pred, mask, split, it):
        if self.args.viz_folder == "":
            save_folder = "tracker/viz/%s_%d" % (self.args.exp_name, self.args.val_epoch)
        else:
            save_folder = "%s/%s_%d" % (self.args.viz_folder, self.args.exp_name, self.args.val_epoch)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        self.viz_attention_weights(self.act_att_weights, self.obs_att_weights, self.seqs, self.seq_lens, split, it, save_folder)

        all_floorplan_images = []
        for im in self.floorplan_images:
            all_floorplan_images.extend([im for t in range(self.args.timesteps)])
        self.floorplan_images = all_floorplan_images  # len: batch_size * timesteps

        self.goal_map = F.interpolate(self.goal_map.unsqueeze(1), scale_factor=self.args.debug_scale).squeeze(1)
        self.path_map = F.interpolate(self.path_map.unsqueeze(1), scale_factor=self.args.debug_scale).squeeze(1)
        goal_pred = F.interpolate(goal_pred.unsqueeze(1), scale_factor=self.args.debug_scale).squeeze(1)
        path_pred = F.interpolate(path_pred.unsqueeze(1), scale_factor=self.args.debug_scale).squeeze(1)
        mask = F.interpolate(mask.unsqueeze(1), scale_factor=self.args.debug_scale).squeeze(1)

        goal_pred = utils.minmax_normalize(goal_pred)
        path_pred = utils.minmax_normalize(path_pred)

        map_masks = viz_utils.get_masks(self.floorplan_images, mask, goal_pred)
        target_images = viz_utils.get_floorplan_with_goal_path_maps(self.floorplan_images,
                            self.goal_map, self.path_map, scale_factor=self.args.debug_scale,
                            target=True)
        predicted_images = viz_utils.get_floorplan_with_goal_path_maps(self.floorplan_images,
                            goal_pred, None, scale_factor=self.args.debug_scale)

        path_pred = path_pred.reshape(self.args.batch_size * self.args.timesteps, self.args.max_steps-1, 3, path_pred.shape[-2], path_pred.shape[-1])

        path_pred = (255*torch.flip(path_pred, [3])).type(torch.ByteTensor).cpu().numpy()

        new_belief = path_pred[:, :, 0, :, :]
        obs_likelihood = path_pred[:, :, 1, :, :]
        belief = path_pred[:, :, 2, :, :]

        viz_utils.save_floorplans_with_belief_maps([map_masks, target_images, predicted_images],
            [new_belief, obs_likelihood, belief], self.args.max_steps-1, self.args.batch_size, self.args.timesteps, split, it, save_folder=save_folder)

    def get_predictions(self, seq, seq_mask, seq_lens, batch, xyzhe, simulator_next_action):

        all_masks = []
        spatial_map, mask = self.mapper.init_map(xyzhe)

        N = seq.shape[0]
        T = self.args.timesteps

        self.act_att_weights = torch.zeros(N, T, self.args.max_steps-1, torch.max(seq_lens))
        self.obs_att_weights = torch.zeros(N, T, self.args.max_steps-1, torch.max(seq_lens))

        goal_pred = torch.zeros(N, T, self.args.map_range_y, self.args.map_range_x, device=spatial_map.device)
        path_pred = torch.zeros(N, T, self.args.max_steps-1, 3, self.args.map_range_y, self.args.map_range_x, device=spatial_map.device)

        if self.args.debug_mode:
            floor_maps = self.floor_maps(self.sim.getState())

        for t in range(self.args.timesteps):
            rgb, depth, states = self.sim.getPanos()
            spatial_map, mask, ftm = self.mapper(rgb, depth, states, spatial_map, mask)

            belief = self.mapper.belief_map(xyzhe, self.args.filter_input_sigma).log()
            if t == 0:
                belief_pred = torch.zeros(N, T, self.args.max_steps, self.args.heading_states, belief.shape[-2], belief.shape[-1], device=spatial_map.device)
            belief_pred[:, t, 0, :, :, :] = belief.exp()
            state = None
            for k in range(self.args.max_steps-1):
                input_belief = belief
                new_belief,obs_likelihood,state,act_att_weights,obs_att_weights,_ = self.model(k, seq, seq_mask, seq_lens, belief, spatial_map, state)
                belief = new_belief + obs_likelihood

                self.act_att_weights[:, t, k, :] = act_att_weights
                self.obs_att_weights[:, t, k, :] = obs_att_weights

                # Renormalize
                belief = belief - belief.reshape(belief.shape[0], -1).logsumexp(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                path_pred[:, t, k, 0, :, :] = F.interpolate(new_belief.exp().sum(dim=1).unsqueeze(1),
                                        scale_factor=self.args.belief_downsample_factor).squeeze(1)

                path_pred[:, t, k, 1, :, :] = F.interpolate(obs_likelihood.exp().sum(dim=1).unsqueeze(1),
                                        scale_factor=self.args.belief_downsample_factor).squeeze(1)

                path_pred[:, t, k, 2, :, :] = F.interpolate(belief.exp().sum(dim=1).unsqueeze(1),
                                        scale_factor=self.args.belief_downsample_factor).squeeze(1)

                belief_pred[:, t, k+1, :, :, :] = belief.exp()

            if self.args.debug_mode:

                base_maps = self.overlay_mask(floor_maps, mask)
                belief_maps = self.overlay_belief(base_maps, belief_pred[:,t])

                cv2.imshow('belief', belief_maps[0])
                cv2.waitKey(0)

            goal_pred[:,t,:,:] = F.interpolate(belief.exp().sum(dim=1).unsqueeze(1),
                                    scale_factor=self.args.belief_downsample_factor).squeeze(1)

            all_masks.append(mask)
            simulator_next_action()

        mask = torch.cat(all_masks, dim=1).flatten(0, 1)

        self.seq_lens = seq_lens
        self.seqs = seq

        return goal_pred.flatten(0,1), path_pred.flatten(0,3), mask


if __name__ == "__main__":

    args = cfg.parse_args()
    args_text = cfg.print_args(args)

    if args.eval is False:
        if args.start_epoch != 1:
            filepath = "%s/%s/%s_%d" % (args.snapshot_dir, args.exp_name, args.exp_name, args.start_epoch-1)
        else:
            filepath = None
        trainer = FilterTrainer(args=args, filepath=filepath)
        trainer.train()
    else:
        filepath = "%s/%s/%s_%d" % (args.snapshot_dir, args.exp_name, args.exp_name, args.val_epoch)
        evaluator = FilterEvaluator(filepath=filepath, args=args)
        evaluator.evaluate("val_unseen")
        evaluator.evaluate("val_seen")
