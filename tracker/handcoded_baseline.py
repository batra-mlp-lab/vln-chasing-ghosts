import math
import torch
from tqdm import tqdm

import cfg
import utils
from trainer import Trainer
from evaluator import Evaluator
from modules.mapper import Mapper


class DistanceOptimizer(Trainer):

    def __init__(self, args):
        args.eval = False
        args.preloading = False
        super(DistanceOptimizer, self).__init__(args, filepath=None)

    def calculate_avg_distance(self):
        distance = 0.0
        iterations = int(math.ceil(len(self.traindata.data) / float(self.args.batch_size)))
        last_batch_valid_idx = len(self.traindata.data) - (iterations-1)*self.args.batch_size

        for it in tqdm(range(iterations)):
            valid_batch_len = last_batch_valid_idx if it == iterations - 1 else self.args.batch_size

            seq, seq_mask, seq_lens, batch = self.traindata.get_batch()
            self.sim.newEpisode(batch)
            xyzhe = self.sim.getXYZHE()
            goal_xy = self.traindata.goal_coords()
            dist = utils.euclidean_distance(xyzhe[:, :2], goal_xy)
            distance += dist[:valid_batch_len].cpu().numpy().sum()

        avg_distance = distance / len(self.traindata.data)
        print("Average distance from start to goal position in train split: %0.4f metres" % avg_distance)
        print("For gridcellsize: %f metres/pixel, it should be %0.4f pixels" % (self.args.gridcellsize, avg_distance/self.args.gridcellsize))


class HandCodedBaselineEvaluator(Evaluator):

    def __init__(self, args, filepath=None):
        args.eval = True
        super(HandCodedBaselineEvaluator, self).__init__(args, filepath=filepath)

        radius_in_pixels = int(self.args.handcoded_radius / self.args.gridcellsize)
        self.avg_dist_goal_mask = utils.get_gaussian_blurred_map_mask(
                                    self.args.map_range_y, self.args.map_range_x, radius_in_pixels,
                                    (self.args.blur_kernel_x, self.args.blur_kernel_y))

    def init_models(self):
        self.args.belief_downsample_factor = 1
        self.mapper = Mapper(self.args).to(self.args.device)

    def load_model_weights(self):
        pass

    def set_models_eval(self):
        self.mapper.eval()

    def get_predictions(self, seq, seq_mask, seq_lens, batch, xyzhe, simulator_next_action):

        all_masks = []

        spatial_map, mask = self.mapper.init_map(xyzhe)

        for t in range(self.args.timesteps):
            rgb, depth, states = self.sim.getPanos()
            spatial_map, mask, ftm = self.mapper(rgb, depth, states, spatial_map, mask)

            all_masks.append(mask)
            if self.args.timesteps != 1:
                simulator_next_action()

        mask = torch.cat(all_masks, dim=1).flatten(0, 1)

        goal_pred = mask * self.avg_dist_goal_mask

        path_pred = None

        return goal_pred, path_pred, mask


if __name__ == "__main__":

    args = cfg.parse_args()
    args_text = cfg.print_args(args)

    # Comment next 2 lines if you just want to run evaluation for some radius
    trainer = DistanceOptimizer(args=args)
    trainer.calculate_avg_distance()

    evaluator = HandCodedBaselineEvaluator(args=args, filepath=None)
    evaluator.evaluate("val_seen")
    evaluator.evaluate("val_unseen")
