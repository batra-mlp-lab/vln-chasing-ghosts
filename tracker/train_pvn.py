import torch
import torch.nn as nn
from torch import optim

import os
import time
import cv2

import utils
import cfg
from trainer import Trainer
from evaluator import Evaluator
from modules.mapper import Mapper
from modules.navigators import UnetNavigator
from modules.loss import MaskedMSELoss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PVNBaselineTrainer(Trainer):
    """ Train the PVN Baseline """

    def __init__(self, args):
        super(PVNBaselineTrainer, self).__init__(args, filepath=None)

        self.mapper = Mapper(self.args).to(self.args.device)
        self.navigator = UnetNavigator(self.args).to(self.args.device)
        self.navigator.init_weights()

        self.goal_criterion = MaskedMSELoss()
        self.path_criterion = MaskedMSELoss()

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

            for it in range(self.args.validation_iterations):

                # Load minibatch and simulator
                seq, seq_mask, seq_lens, batch = val_dataloader.get_batch()
                self.sim.newEpisode(batch)

                if self.args.multi_maps:
                    all_spatial_maps = []
                    all_masks = []

                # Run the mapper for one step
                xyzhe = self.sim.getXYZHE()
                spatial_map,mask = self.mapper.init_map(xyzhe)
                for t in range(self.args.timesteps):
                    rgb,depth,states = self.sim.getPanos()
                    spatial_map,mask,ftm = self.mapper(rgb,depth,states,spatial_map,mask)
                    if self.args.multi_maps:
                        all_spatial_maps.append(spatial_map.unsqueeze(1))
                        all_masks.append(mask.unsqueeze(1))
                    if self.args.timesteps != 1:
                        self.sim.takePseudoSupervisedAction(val_dataloader.get_supervision)

                if self.args.multi_maps:
                    spatial_map = torch.cat(all_spatial_maps, dim=1).flatten(0, 1)
                    mask = torch.cat(all_masks, dim=1).flatten(0, 1)

                    seq = seq.unsqueeze(1).expand(-1, self.args.timesteps, -1).flatten(0, 1)
                    seq_lens = seq_lens.unsqueeze(1).expand(-1, self.args.timesteps).flatten(0, 1)

                # Predict with unet
                pred = self.navigator(seq, seq_lens, spatial_map)
                goal_pred = pred[:,0,:,:]
                path_pred = pred[:,1,:,:]

                goal_map = self.mapper.heatmap(val_dataloader.goal_coords(), self.args.goal_heatmap_sigma)
                path_map = self.mapper.heatmap(val_dataloader.path_coords(), self.args.path_heatmap_sigma)

                goal_map_max, _ = torch.max(goal_map, dim=2, keepdim=True)
                goal_map_max, _ = torch.max(goal_map_max, dim=1, keepdim=True)
                goal_map /= goal_map_max

                path_map_max, _ = torch.max(path_map, dim=2, keepdim=True)
                path_map_max, _ = torch.max(path_map_max, dim=1, keepdim=True)
                path_map /= path_map_max

                if self.args.belief_downsample_factor > 1:
                    goal_map = nn.functional.avg_pool2d(goal_map, kernel_size=self.args.belief_downsample_factor, stride=self.args.belief_downsample_factor)
                    path_map = nn.functional.avg_pool2d(path_map, kernel_size=self.args.belief_downsample_factor, stride=self.args.belief_downsample_factor)
                    mask = nn.functional.max_pool2d(mask, kernel_size=self.args.belief_downsample_factor, stride=self.args.belief_downsample_factor)

                if self.args.multi_maps:
                    goal_map = goal_map.unsqueeze(1).expand(-1, self.args.timesteps, -1, -1).flatten(0, 1)
                    path_map = path_map.unsqueeze(1).expand(-1, self.args.timesteps, -1, -1).flatten(0, 1)

                loss = self.goal_criterion(goal_pred,goal_map) + self.path_criterion(path_pred,path_map)

                total_loss += (1.0 / self.args.validation_iterations) * loss

        return total_loss

    def logging_loop(self, it):
        """ Logging and checkpointing stuff """

        if it % self.args.validate_every == 0:

            self.mapper.eval()
            self.navigator.eval()

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
            self.navigator.train()

        elif it % self.args.log_every == 0:
            self.prev_time, time_taken = utils.time_it(self.prev_time)
            print("Iteration: %d Loss: %f Time: %0.2f secs" % (it, self.loss.item(), time_taken))

            if self.visdom:
                self.visdom.line(it, self.loss.item(), "train_loss", "Train Loss", "Iterations", "Loss", title="Train Phase")

        if it % self.args.checkpoint_every == 0:
            saver = {"mapper": self.mapper.state_dict(),
                     "navigator": self.navigator.state_dict(),
                     "args": self.args}
            dir = "%s/%s" % (self.args.snapshot_dir, self.args.exp_name)
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(
                saver, "%s/%s_%d" % (dir, self.args.exp_name, it)
            )
            if self.visdom:
                self.visdom.save()

    def plot_grad_flow(self, named_parameters):
        ave_grads = []
        layers = []

        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.savefig("gradient.png")

    def train(self):
        """
        Supervised training of the mapper and navigator
        """

        self.prev_time = time.time()

        map_opt = optim.Adam(self.mapper.parameters(), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay)
        nav_opt = optim.Adam(self.navigator.parameters(), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay)

        # Set models to train phase
        self.mapper.train()
        self.navigator.train()

        for it in range(1, self.args.max_iterations + 1):
            map_opt.zero_grad()
            nav_opt.zero_grad()

            # Load minibatch and simulator
            seq, seq_mask, seq_lens, batch = self.traindata.get_batch()
            self.sim.newEpisode(batch)

            if self.args.multi_maps:
                all_spatial_maps = []
                all_masks = []

            # Run the mapper for multiple timesteps and keep each map
            xyzhe = self.sim.getXYZHE()
            spatial_map,mask = self.mapper.init_map(xyzhe)
            for t in range(self.args.timesteps):
                rgb,depth,states = self.sim.getPanos()
                spatial_map,mask,ftm = self.mapper(rgb,depth,states,spatial_map,mask)
                if self.args.multi_maps:
                    all_spatial_maps.append(spatial_map.unsqueeze(1))
                    all_masks.append(mask.unsqueeze(1))
                if self.args.timesteps != 1:
                    self.sim.takePseudoSupervisedAction(self.traindata.get_supervision)

            if self.args.multi_maps:
                spatial_map = torch.cat(all_spatial_maps, dim=1).flatten(0, 1)
                mask = torch.cat(all_masks, dim=1).flatten(0, 1)

                seq = seq.unsqueeze(1).expand(-1, self.args.timesteps, -1).flatten(0, 1)
                seq_lens = seq_lens.unsqueeze(1).expand(-1, self.args.timesteps).flatten(0, 1)

            # Predict with unet
            pred = self.navigator(seq, seq_lens, spatial_map)

            goal_pred = pred[:,0,:,:]
            path_pred = pred[:,1,:,:]

            goal_map = self.mapper.heatmap(self.traindata.goal_coords(), self.args.goal_heatmap_sigma)
            path_map = self.mapper.heatmap(self.traindata.path_coords(), self.args.path_heatmap_sigma)

            goal_map_max, _ = torch.max(goal_map, dim=2, keepdim=True)
            goal_map_max, _ = torch.max(goal_map_max, dim=1, keepdim=True)
            goal_map /= goal_map_max

            path_map_max, _ = torch.max(path_map, dim=2, keepdim=True)
            path_map_max, _ = torch.max(path_map_max, dim=1, keepdim=True)
            path_map /= path_map_max

            if self.args.belief_downsample_factor > 1:
                goal_map = nn.functional.avg_pool2d(goal_map, kernel_size=self.args.belief_downsample_factor, stride=self.args.belief_downsample_factor)
                path_map = nn.functional.avg_pool2d(path_map, kernel_size=self.args.belief_downsample_factor, stride=self.args.belief_downsample_factor)
                mask = nn.functional.max_pool2d(mask, kernel_size=self.args.belief_downsample_factor, stride=self.args.belief_downsample_factor)

            if self.args.multi_maps:
                goal_map = goal_map.unsqueeze(1).expand(-1, self.args.timesteps, -1, -1).flatten(0, 1)
                path_map = path_map.unsqueeze(1).expand(-1, self.args.timesteps, -1, -1).flatten(0, 1)

            self.loss = self.goal_criterion(goal_pred,goal_map) + self.path_criterion(path_pred,path_map)

            self.loss.backward()
            map_opt.step()
            nav_opt.step()
            self.logging_loop(it)


class PVNBaselineEvaluator(Evaluator):
    def __init__(self, args, filepath):
        super(PVNBaselineEvaluator, self).__init__(args, filepath)

    def init_models(self):
        self.args.belief_downsample_factor = 1

        self.mapper = Mapper(self.args).to(self.args.device)
        self.navigator = UnetNavigator(self.args).to(self.args.device)

    def load_model_weights(self):
        self.mapper.load_state_dict(self.loader["mapper"])
        self.navigator.load_state_dict(self.loader["navigator"])

    def set_models_eval(self):
        self.mapper.eval()
        self.navigator.eval()

    def get_predictions(self, seq, seq_mask, seq_lens, batch, xyzhe, simulator_next_action):

        if self.args.multi_maps:
            all_spatial_maps = []
            all_masks = []

        spatial_map, mask = self.mapper.init_map(xyzhe)

        for t in range(self.args.timesteps):
            rgb, depth, states = self.sim.getPanos()
            spatial_map, mask, ftm = self.mapper(rgb, depth, states, spatial_map, mask)
            if self.args.multi_maps:
                all_spatial_maps.append(spatial_map.unsqueeze(1))
                all_masks.append(mask)
            if self.args.timesteps != 1:
                simulator_next_action()

        if self.args.multi_maps:
            spatial_map = torch.cat(all_spatial_maps, dim=1).flatten(0, 1)
            mask = torch.cat(all_masks, dim=1).flatten(0, 1)

            seq = seq.unsqueeze(1).expand(-1, self.args.timesteps, -1).flatten(0, 1)
            seq_lens = seq_lens.unsqueeze(1).expand(-1, self.args.timesteps).flatten(0, 1)

        # Predict with unet
        pred = self.navigator(seq, seq_lens, spatial_map)

        # shape: (batch_size*timesteps, 2)
        goal_pred = pred[:, 0, :, :]
        path_pred = pred[:, 1, :, :]

        return goal_pred, path_pred, mask


if __name__ == "__main__":

    args = cfg.parse_args()
    args_text = cfg.print_args(args)

    if args.eval is False:
        trainer = PVNBaselineTrainer(args=args)
        trainer.train()
    else:
        filepath = "%s/%s/%s_%d" % (args.snapshot_dir, args.exp_name, args.exp_name, args.val_epoch)
        trainer = PVNBaselineEvaluator(filepath=filepath, args=args)
        trainer.evaluate("val_seen")
        trainer.evaluate("val_unseen")
