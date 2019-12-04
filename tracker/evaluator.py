from abc import abstractmethod

import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from trainer import Trainer
from modules.mapper import Mapper
import metrics
import utils


class Evaluator(Trainer):
    """ Generic class for running evaluations for different models on the goal prediction task """

    def __init__(self, args=None, filepath=None, load_sim=True):
        super(Evaluator, self).__init__(args, filepath, load_sim=load_sim)
        self.init_models()
        self.load_model_weights()

    @abstractmethod
    def init_models(self):
        """ Initialize the model """
        raise NotImplementedError

    @abstractmethod
    def load_model_weights(self):
        """ Load the pretrained model weights """
        raise NotImplementedError

    @abstractmethod
    def set_models_eval(self):
        """ Set models to eval phase """
        raise NotImplementedError

    @abstractmethod
    def get_predictions(self, seq, seq_mask, seq_lens, batch, xyzhe, simulator_next_action):
        """ Get predictions from the model
        Args:
            simulator_next_action: Function which takes next action in the simulator
        Return:
            goal_pred: FloatTensor, [N*timesteps, belief_map_y, belief_map_x]
            path_pred: FloatTensor, [N*timesteps, belief_map_y, belief_map_x]
            mask: FloatTensor with values from {0,1}, [N*timesteps, belief_map_y, belief_map_x]
        """
        raise NotImplementedError

    def eval_logging(self, split, timestep_nav_error, average_nav_error, timestep_success_rate, average_success_rate, timestep_map_coverage, average_map_coverage, timestep_goal_seen, average_goal_seen):

        timestep_nav_error = np.array(timestep_nav_error).mean(axis=1)
        average_nav_error = np.array(average_nav_error).mean()
        timestep_success_rate = 100 * np.array(timestep_success_rate).mean(axis=1)
        average_success_rate = 100 * np.array(average_success_rate).mean()
        timestep_map_coverage = 100 * np.array(timestep_map_coverage).mean(axis=1)
        average_map_coverage = 100 * np.array(average_map_coverage).mean()
        timestep_goal_seen = 100 * np.array(timestep_goal_seen).mean(axis=1)
        average_goal_seen = 100 * np.array(average_goal_seen).mean()

        map_size_in_m2 = (self.args.map_range_x*self.args.gridcellsize) * (self.args.map_range_y*self.args.gridcellsize)
        timestep_map_coverage_in_m2 = (timestep_map_coverage / 100.0) * map_size_in_m2
        average_map_coverage_in_m2 = (average_map_coverage / 100.0) * map_size_in_m2

        result_str = "Epoch %d Split %s scores: Nav Error=%0.4f Success Rate=%0.4f Map Coverage=%0.4f (%0.4f metres^2) Goal Seen Rate=%0.4f" %(self.args.val_epoch, split, average_nav_error, average_success_rate, average_map_coverage, average_map_coverage_in_m2, average_goal_seen)
        print(result_str)

        timestep_result_str = ""
        for t in range(self.args.timesteps):
            r_str = "Epoch %d Split %s Timestep: %d Scores: Nav Error=%0.4f Success Rate=%0.4f Map Coverage=%0.4f (%0.4f metres^2) Goal Seen Rate=%0.4f" %(self.args.val_epoch, split, t, timestep_nav_error[t], timestep_success_rate[t], timestep_map_coverage[t], timestep_map_coverage_in_m2[t], timestep_goal_seen[t])
            timestep_result_str += r_str + "\n"
        print(timestep_result_str)

        if self.visdom:
            self.visdom.line(self.args.val_epoch, average_nav_error, "nav_error", "%s Avg Nav Error" % split, "Epochs", "Navigation Error", title="Average Nav Error - Val Split")
            self.visdom.line(self.args.val_epoch, average_success_rate, "success_rate", "%s Avg Success Rate" % split, "Epochs", "Success Rate", title="Average Success Rate - Val Split")
            self.visdom.line(self.args.val_epoch, average_map_coverage, "map_coverage", "%s Avg Map Coverage" % split, "Epochs", "Map Coverage", title="Average Map Coverage - Val Split")
            self.visdom.line(self.args.val_epoch, average_map_coverage_in_m2, "map_coverage_in_metres", "%s Avg Map Coverage in Metres^2" % split, "Epochs", "Map Coverage in Metres^2", title="Average Map Coverage (in metres^2)- Val Split")
            self.visdom.line(self.args.val_epoch, average_goal_seen, "goal_seen", "%s Avg Goal Seen Rate" % split, "Epochs", "Goal Seen Rate", title="Average Goal Seen Rate - Val Split")
            for t in range(self.args.timesteps):
                self.visdom.line(self.args.val_epoch, timestep_nav_error[t], "nav_error-t%d" % t, "%s Nav Error-t%d" % (split, t), "Epochs", "Navigation Error", title="Nav Error Timestep %d - Val Split" % t)
                self.visdom.line(self.args.val_epoch, timestep_success_rate[t], "success_rate-t%d" % t, "%s Success Rate-t%d" % (split, t), "Epochs", "Success Rate", title="Success Rate Timestep %d - Val Split" % t)
                self.visdom.line(self.args.val_epoch, timestep_map_coverage[t], "map_coverage-t%d" % t, "%s Map Coverage-t%d" % (split, t), "Epochs", "Map Coverage", title="Map Coverage Timestep %d - Val Split" % t)
                self.visdom.line(self.args.val_epoch, timestep_map_coverage_in_m2[t], "map_coverage-metres-t%d" % t, "%s Map Coverage (in Metres^2)-t%d" % (split, t), "Epochs", "Map Coverage in Metres^2", title="Map Coverage (in Metres^2) Timestep %d - Val Split" % t)
                self.visdom.line(self.args.val_epoch, timestep_goal_seen[t], "goal_seen_rate-t%d" % t, "%s Goal Seen Rate-t%d" % (split, t), "Epochs", "Goal Seen Rate", title="Goal Seen Rate Timestep %d - Val Split" % t)
            self.visdom.text(result_str)
            self.visdom.text(timestep_result_str.replace("\n", "<br>"))
            self.visdom.save()

    def simulator_next_action(self):
        self.sim.takePseudoSupervisedAction(self.dataloader.get_supervision)

    def evaluate(self, split):

        self.set_models_eval()

        with torch.no_grad():
            if split == "val_seen":
                self.dataloader = self.valseendata
            elif split == "val_unseen":
                self.dataloader = self.valunseendata

            iterations = int(math.ceil(len(self.dataloader.data) / float(self.args.batch_size)))
            last_batch_valid_idx = len(self.dataloader.data) - (iterations-1)*self.args.batch_size

            timestep_nav_error = [[] for i in range(self.args.timesteps)]
            timestep_success_rate = [[] for i in range(self.args.timesteps)]
            average_nav_error = []
            average_success_rate = []

            timestep_map_coverage = [[] for i in range(self.args.timesteps)]
            timestep_goal_seen = [[] for i in range(self.args.timesteps)]
            average_map_coverage = []
            average_goal_seen = []

            mapper = Mapper(self.args).to(self.args.device)

            for it in tqdm(range(iterations), desc="Evaluation Progress for %s split" % split):

                valid_batch_len = last_batch_valid_idx if it == iterations - 1 else self.args.batch_size

                seq, seq_mask, seq_lens, batch = self.dataloader.get_batch()

                self.sim.newEpisode(batch)
                self.floorplan_images = utils.get_floorplan_images(self.sim.getState(),
                                        self.floorplan, self.args.map_range_x, self.args.map_range_y,
                                        scale_factor=self.args.debug_scale)

                xyzhe = self.sim.getXYZHE()
                mapper.init_map(xyzhe)
                goal_pos = self.dataloader.get_goal_coords_on_map_grid(mapper.map_center)

                pred_goal_map, pred_path_map, mask = self.get_predictions(seq, seq_mask, seq_lens, batch, xyzhe,
                                                                          self.simulator_next_action)

                # shape: (batch_size, 2)
                self.goal_map = mapper.heatmap(self.dataloader.goal_coords(), self.args.goal_heatmap_sigma)
                self.path_map = mapper.heatmap(self.dataloader.path_coords(), self.args.path_heatmap_sigma)

                if self.args.multi_maps:
                    # shape: (batch_size*timesteps, 2)
                    self.goal_map = self.goal_map.unsqueeze(1).expand(-1, self.args.timesteps, -1, -1).flatten(0, 1)
                    self.path_map = self.path_map.unsqueeze(1).expand(-1, self.args.timesteps, -1, -1).flatten(0, 1)

                # shape: (batch_size*timesteps, 2)
                goal_pred_argmax = utils.compute_argmax(pred_goal_map).flip(1)

                # shape: (batch_size, timesteps, 2)
                goal_pred_argmax = goal_pred_argmax.reshape(-1, self.args.timesteps, 2)
                goal_pred_xy = self.dataloader.convert_map_pixels_to_xy_coords(goal_pred_argmax,
                                            mapper.map_center, multi_timestep_input=True)

                # shape: (batch_size, 2)
                goal_target_xy = self.dataloader.goal_coords()
                # shape: (batch_size, timesteps, 2)
                goal_target_xy = goal_target_xy.unsqueeze(1).expand(-1, self.args.timesteps, -1)

                # shape: (batch_size, timesteps, map_range_y, map_range_x)
                b_t_mask = mask.reshape(-1, self.args.timesteps, self.args.map_range_y, self.args.map_range_x)

                batch_timestep_map_coverage, batch_average_map_coverage = \
                    metrics.map_coverage(b_t_mask)
                batch_timestep_goal_seen, batch_average_goal_seen = \
                    metrics.goal_seen_rate(b_t_mask, goal_pos, self.args)
                batch_timestep_nav_error, batch_average_nav_error = \
                    metrics.nav_error(goal_target_xy, goal_pred_xy, self.args)
                batch_timestep_success_rate, batch_average_success_rate = \
                    metrics.success_rate(goal_target_xy, goal_pred_xy, self.args)

                for n in range(valid_batch_len):
                    average_nav_error.append(batch_average_nav_error[n].item())
                    average_success_rate.append(batch_average_success_rate[n].item())
                    average_map_coverage.append(batch_average_map_coverage[n].item())
                    average_goal_seen.append(batch_average_goal_seen[n].item())

                    for t in range(batch_timestep_map_coverage.shape[1]):
                        timestep_nav_error[t].append(batch_timestep_nav_error[n][t].item())
                        timestep_success_rate[t].append(batch_timestep_success_rate[n][t].item())
                        timestep_map_coverage[t].append(batch_timestep_map_coverage[n][t].item())
                        timestep_goal_seen[t].append(batch_timestep_goal_seen[n][t].item())

            self.eval_logging(split, timestep_nav_error, average_nav_error, timestep_success_rate, average_success_rate, timestep_map_coverage, average_map_coverage, timestep_goal_seen, average_goal_seen)
