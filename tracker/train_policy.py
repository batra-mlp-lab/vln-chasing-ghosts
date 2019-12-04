import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
torch.set_printoptions(sci_mode=False)

import os
import time
import math
import json
import cv2

import utils
import viz_utils
import cfg
from trainer import Trainer
from simulator import PanoSimulatorWithGraph
from dataloader import DataLoader
from vln_eval import Evaluation
from modules.mapper import Mapper
from modules.navigators import Filter
from modules.policy import ReactivePolicy
from modules.loss import LogBeliefLoss

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def belief_at_nodes(belief, nodes, sigma, gridcellsize):
    # Accumulate for policy - log prob or prob? Include heading bins or not? Perhaps some batchnorm
    # belief (N, C, H, W)
    # nodes (P, 4) - batch_index, rel_x, rel_y, rel_z
    # return (P)

    # world to map coords
    P = nodes.shape[0]
    out_size = torch.tensor([belief.shape[-1], belief.shape[-2]], device=belief.device).float()
    coords = (nodes[:,1:3]/gridcellsize + 0.5*out_size).unsqueeze(1).unsqueeze(1) # P, 1, 1, 2

    # heatmap masks - generate bivariate Gaussians centered at coords
    mask = torch.zeros(P, belief.shape[0], belief.shape[-2], belief.shape[-1], device=belief.device) # P, N, Y, X
    x = torch.arange(start=0,end=belief.shape[-1], device=belief.device).float().unsqueeze(0).unsqueeze(0) # 1, 1, X
    y = torch.arange(start=0,end=belief.shape[-2], device=belief.device).float().unsqueeze(1).unsqueeze(0) # 1, Y, 1
    heat = torch.exp(-0.5 * (((x - coords[:,:,:,0])/sigma)**2 + ((y - coords[:,:,:,1])/sigma)**2 )) # P, Y, X
    mask[torch.arange(P),nodes[:,0].long()] = heat
    bel = belief.sum(dim=1).unsqueeze(0) # 1, N, Y, X
    node_belief = (bel * mask).reshape(P,-1).sum(dim=1)
    return node_belief


class PolicyTrainer(Trainer):
    """ Train a filter and a policy """

    def __init__(self, args, filepath=None):
        super(PolicyTrainer, self).__init__(args, filepath, load_sim=False)
        self.sim = PanoSimulatorWithGraph(self.args)

        # load models
        self.mapper = Mapper(self.args).to(self.args.device)
        self.model = Filter(self.args).to(self.args.device)
        self.policy = ReactivePolicy(self.args).to(self.args.device)

        if filepath:
            loader = torch.load(filepath)
            self.mapper.load_state_dict(loader["mapper"])
            self.model.load_state_dict(loader["filter"])
            self.policy.load_state_dict(loader["policy"])
            print("Loaded Mapper, Filter and Policy from: %s" % filepath)
        elif args:
            self.model.init_weights()
            self.policy.init_weights()
        self.belief_criterion = LogBeliefLoss()
        self.policy_criterion = torch.nn.NLLLoss(ignore_index=self.args.action_ignore_index, reduction='none')


    def validate(self, split):
        """
        split: "val_seen" or "val_unseen"
        """
        vln_eval = Evaluation(split)
        self.sim.record_traj(True)
        with torch.no_grad():

            if(split=="val_seen"):
                val_dataloader = self.valseendata
            elif(split=="val_unseen"):
                val_dataloader = self.valunseendata
            elif(split=="train"):
                val_dataloader = self.traindata

            for it in range(self.args.validation_iterations):

                # Load minibatch and simulator
                seq, seq_mask, seq_lens, batch = val_dataloader.get_batch()
                self.sim.newEpisode(batch)

                # Initialize the mapper
                xyzhe = self.sim.getXYZHE()
                spatial_map,mask = self.mapper.init_map(xyzhe)

                ended = torch.zeros(self.args.batch_size, device=self.args.device).byte()
                for t in range(self.args.timesteps):
                    if self.args.policy_gt_belief:
                        path_xyzhe, path_len = val_dataloader.path_xyzhe()
                    else:
                        rgb,depth,states = self.sim.getPanos()
                        spatial_map,mask,ftm = self.mapper(rgb,depth,states,spatial_map,mask)
                        del states
                        del depth
                        del ftm
                        del rgb

                    features,_,_ = self.sim.getGraphNodes()
                    P = features.shape[0] # num graph nodes
                    belief_features = torch.empty(P, self.args.max_steps, device=self.args.device)

                    state = None
                    steps = self.args.max_steps-1

                    belief = self.mapper.belief_map(xyzhe, self.args.filter_input_sigma).log()
                    sigma = self.args.filter_heatmap_sigma
                    gridcellsize = self.args.gridcellsize*self.args.belief_downsample_factor
                    belief_features[:,0] = belief_at_nodes(belief.exp(), features[:,:4], sigma, gridcellsize)
                    for k in range(steps):
                        if self.args.policy_gt_belief:
                            target_heatmap = self.mapper.belief_map(path_xyzhe[k+1], self.args.filter_heatmap_sigma)
                            belief_features[:,k+1] = belief_at_nodes(target_heatmap, features[:,:4], sigma, gridcellsize)
                        else:
                            input_belief = belief
                            # Train a filter
                            new_belief,obs_likelihood,state,_,_,_ = self.model(k, seq, seq_mask, seq_lens, input_belief, spatial_map, state)
                            belief = new_belief + obs_likelihood
                            # Renormalize
                            belief = belief - belief.reshape(belief.shape[0], -1).logsumexp(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)

                            belief_features[:,k+1] = belief_at_nodes(belief.exp(), features[:,:4], sigma, gridcellsize)

                    # Probs from policy
                    aug_features = torch.cat([features,belief_features], dim=1)
                    log_prob = self.policy(aug_features)

                    # Take argmax action in the sim
                    _,action_idx = log_prob.exp().max(dim=1)
                    ended |= self.sim.takeMultiStepAction(action_idx)

                    if ended.all():
                        break

        # Eval
        scores = vln_eval.score(self.sim.traj, check_all_trajs=False)
        self.sim.record_traj(False)
        return scores['result'][0][split]


    def logging_loop(self, it):
        """ Logging and checkpointing stuff """

        if it % self.args.validate_every == 0:
            self.mapper.eval()
            self.model.eval()
            self.policy.eval()
            scores_train = self.validate("train")
            scores_val_seen = self.validate("val_seen")
            scores_val_unseen = self.validate("val_unseen")

            self.prev_time, time_taken = utils.time_it(self.prev_time)

            print("Iteration: %d Loss: %f Train Success: %f Val Seen Success: %f Val Unseen Success: %f Time: %0.2f secs"
                    %(it, self.loss.item(), scores_train['success'], scores_val_seen['success'], scores_val_unseen['success'], time_taken))

            if self.visdom:
                # visdom: X, Y, key, line_name, x_label, y_label, fig_title
                self.visdom.line(it, self.loss.item(), "train_loss", "Train Loss", "Iterations", "Loss", title=" Train Phase")

                units = {
                    'length': 'm',
                    'error': 'm',
                    'oracle success': '%',
                    'success': '%',
                    'spl': '%'
                }
                sub = self.args.validation_iterations * self.args.batch_size
                for metric,score in scores_train.items():
                    m = metric.title()
                    self.visdom.line(it, score, metric, "Train (%d)" % sub, "Iterations", units[metric], title=m)
                for metric,score in scores_val_seen.items():
                    m = metric.title()
                    self.visdom.line(it, score, metric, "Val Seen (%d)" % sub, "Iterations", units[metric], title=m)
                for metric,score in scores_val_unseen.items():
                    m = metric.title()
                    self.visdom.line(it, score, metric, "Val Unseen (%d)" % sub, "Iterations", units[metric], title=m)
            self.mapper.train()
            self.model.train()
            self.policy.train()

        elif it % self.args.log_every == 0:
            self.prev_time, time_taken = utils.time_it(self.prev_time)
            print("Iteration: %d Loss: %f Time: %0.2f secs" % (it, self.loss.item(), time_taken))
            if self.visdom:
                self.visdom.line(it, self.loss.item(), "train_loss", "Train Loss", "Iterations", "Loss", title="Train Phase")

        if it % self.args.checkpoint_every == 0:
            saver = {"mapper": self.mapper.state_dict(),
                     "args": self.args,
                     "filter": self.model.state_dict(),
                     "policy": self.policy.state_dict()}
            dir = "%s/%s" % (self.args.snapshot_dir, self.args.exp_name)
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(
                saver, "%s/%s_%d" % (dir, self.args.exp_name, it)
            )
            if self.visdom:
                self.visdom.save()


    def train(self):
        """
        Supervised training of the mapper, filter and policy
        """

        torch.autograd.set_detect_anomaly(True)
        self.prev_time = time.time()

        map_opt = self.optimizer(self.mapper.parameters())
        model_opt = self.optimizer(self.model.parameters())
        policy_opt = self.optimizer(self.policy.parameters())

        # Set models to train phase
        self.mapper.train()
        self.model.train()
        self.policy.train()

        for it in range(1, self.args.max_iterations + 1):
            map_opt.zero_grad()
            model_opt.zero_grad()
            policy_opt.zero_grad()

            traindata = self.traindata

            # Load minibatch and simulator
            seq, seq_mask, seq_lens, batch= traindata.get_batch()
            self.sim.newEpisode(batch)

            # Initialize the mapper
            xyzhe = self.sim.getXYZHE()
            spatial_map,mask = self.mapper.init_map(xyzhe)

            # Note Filter model is being trained to predict the GT path. This is not the path taken by Policy.
            path_xyzhe, path_len = traindata.path_xyzhe()
            self.loss = 0
            ended = torch.zeros(self.args.batch_size, device=self.args.device).byte()
            for t in range(self.args.timesteps):
                if not self.args.policy_gt_belief:
                    rgb,depth,states = self.sim.getPanos()
                    spatial_map,mask,ftm = self.mapper(rgb,depth,states,spatial_map,mask)

                    del states
                    del depth
                    del ftm
                    del rgb

                features,_,_ = self.sim.getGraphNodes()
                P = features.shape[0] # num graph nodes
                belief_features = torch.empty(P, self.args.max_steps, device=self.args.device)

                state = None
                steps = self.args.max_steps-1

                gt_input_belief = None
                belief = self.mapper.belief_map(path_xyzhe[0], self.args.filter_input_sigma).log()
                sigma = self.args.filter_heatmap_sigma
                gridcellsize = self.args.gridcellsize*self.args.belief_downsample_factor
                belief_features[:,0] = belief_at_nodes(belief.exp(), features[:,:4], sigma, gridcellsize)
                for k in range(steps):
                    target_heatmap = self.mapper.belief_map(path_xyzhe[k+1], self.args.filter_heatmap_sigma)
                    if self.args.policy_gt_belief:
                        belief_features[:,k+1] = belief_at_nodes(target_heatmap, features[:,:4], sigma, gridcellsize)
                    else:
                        input_belief = belief
                        # Train a filter
                        if self.args.teacher_force_motion_model:
                            gt_input_belief = self.mapper.belief_map(path_xyzhe[k], self.args.filter_heatmap_sigma).log()
                            new_belief,obs_likelihood,state,_,_,new_gt_belief = self.model(k, seq, seq_mask, seq_lens, input_belief,
                                                                                            spatial_map, state, gt_input_belief)
                            belief = new_belief.detach() + obs_likelihood #Don't backprop through belief
                        else:
                            new_belief,obs_likelihood,state,_,_,new_gt_belief = self.model(k, seq, seq_mask, seq_lens, input_belief,
                                                                                            spatial_map, state, gt_input_belief)
                            belief = new_belief + obs_likelihood
                        # Renormalize
                        belief = belief - belief.reshape(belief.shape[0], -1).logsumexp(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)

                        # Determine target and loss for the filter
                        belief_loss = self.belief_criterion(belief, target_heatmap, valid=~ended) / (~ended).sum()
                        if self.args.teacher_force_motion_model: # Separate loss for the motion model
                            belief_loss += self.belief_criterion(new_gt_belief, target_heatmap, valid=~ended) / (~ended).sum()
                        self.loss += belief_loss

                        belief_features[:,k+1] = belief_at_nodes(belief.exp(), features[:,:4], sigma, gridcellsize)

                # Train a policy
                aug_features = torch.cat([features,belief_features], dim=1)
                log_prob = self.policy(aug_features)

                # Train a policy
                target_idx = traindata.closest_to_goal(self.sim.G)
                policy_loss = self.args.policy_loss_lambda * self.policy_criterion(log_prob, target_idx).sum() / (~ended).sum()
                self.loss += policy_loss

                # Take action in the sim
                if  self.args.supervision_prob < 0:
                    supervision_prob = 1.0 - float(it)/self.args.max_iterations
                else:
                    supervision_prob = self.args.supervision_prob
                sampled_action = D.Categorical(log_prob.exp()).sample()
                weights = torch.tensor([1.0-supervision_prob, supervision_prob], dtype=torch.float, device=log_prob.device)
                ix = torch.multinomial(weights, self.args.batch_size, replacement=True).byte()
                action_idx = torch.where(ix, target_idx, sampled_action)
                ended |= self.sim.takeMultiStepAction(action_idx)

                trunc = self.args.truncate_after
                if (t%trunc)==(trunc-1) or t+1 == self.args.timesteps or ended.all():
                    self.loss.backward()
                    self.loss.detach_()
                    spatial_map.detach_()
                    if state is not None:
                        state = self.model.detach_state(state)
                        state = None # Recalc to get gradients from later in the decoding - #TODO keep this part of the graph without re-running the forward part

                if ended.all():
                    break

            del spatial_map
            del mask

            map_opt.step()
            model_opt.step()
            policy_opt.step()
            self.logging_loop(it)


class PolicyEvaluator(Trainer):

    def __init__(self, args, filepath):
        super(PolicyEvaluator, self).__init__(args, filepath, load_sim=False)

        # load models
        self.mapper = Mapper(self.args).to(self.args.device)
        self.model = Filter(self.args).to(self.args.device)
        self.policy = ReactivePolicy(self.args).to(self.args.device)

        loader = torch.load(filepath)
        self.mapper.load_state_dict(loader["mapper"])
        self.model.load_state_dict(loader["filter"])
        self.policy.load_state_dict(loader["policy"])
        print("Loaded Mapper, Filter and Policy from: %s" % filepath)

        if self.args.viz_folder == "":
            self.args.viz_folder = "tracker/viz/%s_%d" % (self.args.exp_name, self.args.val_epoch)
        else:
            self.args.viz_folder = "%s/%s_%d" % (self.args.viz_folder, self.args.exp_name, self.args.val_epoch)
        if not os.path.exists(self.args.viz_folder) and self.args.viz_eval:
            os.makedirs(self.args.viz_folder)

    def evaluate(self, split):

        sim = PanoSimulatorWithGraph(self.args, disable_rendering=True)
        sim.record_traj(True)

        self.mapper.eval()
        self.model.eval()
        self.policy.eval()

        vln_eval = Evaluation(split)

        with torch.no_grad():
            if split == "val_seen":
                self.dataloader = self.valseendata
            elif split == "val_unseen":
                self.dataloader = self.valunseendata
            else:
                self.dataloader = DataLoader(self.args, splits=[split])

            iterations = int(math.ceil(len(self.dataloader.data) / float(self.args.batch_size)))
            for it in tqdm(range(iterations), desc="Evaluation Progress for %s split" % split):

                # Load minibatch and simulator
                seq, seq_mask, seq_lens, batch= self.dataloader.get_batch()
                sim.newEpisode(batch)

                # Initialize the mapper
                xyzhe = sim.getXYZHE()
                spatial_map,mask = self.mapper.init_map(xyzhe)

                if self.args.viz_eval and it<self.args.viz_iterations:
                    floor_maps = self.floor_maps(sim.getState())

                ended = torch.zeros(self.args.batch_size, device=self.args.device).byte()
                viz_counter = 0
                for t in range(self.args.timesteps):
                    if self.args.policy_gt_belief:
                        path_xyzhe, path_len = self.dataloader.path_xyzhe()
                    else:
                        rgb,depth,states = sim.getPanos()
                        spatial_map,mask,ftm = self.mapper(rgb,depth,states,spatial_map,mask)
                        del states
                        del ftm

                    features,_,_ = sim.getGraphNodes()
                    P = features.shape[0] # num graph nodes
                    belief_features = torch.empty(P, self.args.max_steps, device=self.args.device)

                    state = None
                    steps = self.args.max_steps-1

                    belief = self.mapper.belief_map(xyzhe, self.args.filter_input_sigma).log()
                    if t == 0:
                        belief_pred = torch.zeros(seq.shape[0], self.args.max_steps, self.args.heading_states,
                                                              belief.shape[-2], belief.shape[-1], device=spatial_map.device)
                    belief_pred[:, 0, :, :, :] = belief.exp()

                    sigma = self.args.filter_heatmap_sigma
                    gridcellsize = self.args.gridcellsize*self.args.belief_downsample_factor
                    belief_features[:,0] = belief_at_nodes(belief.exp(), features[:,:4], sigma, gridcellsize)

                    act_att_weights = torch.zeros(seq.shape[0], steps, torch.max(seq_lens))
                    obs_att_weights = torch.zeros(seq.shape[0], steps, torch.max(seq_lens))

                    for k in range(steps):
                        if self.args.policy_gt_belief:
                            target_heatmap = self.mapper.belief_map(path_xyzhe[k+1], self.args.filter_heatmap_sigma)
                            belief_features[:,k+1] = belief_at_nodes(target_heatmap, features[:,:4], sigma, gridcellsize)
                            belief_pred[:, k+1, :, :, :] = target_heatmap
                        else:
                            input_belief = belief
                            # Train a filter
                            new_belief,obs_likelihood,state,act_att,obs_att,_ = self.model(k, seq, seq_mask, seq_lens, input_belief, spatial_map, state)
                            belief = new_belief + obs_likelihood
                            # Renormalize
                            belief = belief - belief.reshape(belief.shape[0], -1).logsumexp(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                            belief_pred[:, k+1, :, :, :] = belief.exp()
                            belief_features[:,k+1] = belief_at_nodes(belief.exp(), features[:,:4], sigma, gridcellsize)

                            act_att_weights[:, k] = act_att
                            obs_att_weights[:, k] = obs_att

                    # Probs from policy
                    aug_features = torch.cat([features,belief_features], dim=1)
                    log_prob = self.policy(aug_features)

                    # Take argmax action in the sim
                    _,action_idx = log_prob.exp().max(dim=1)

                    if self.args.viz_eval and it<self.args.viz_iterations:
                        num_cam_views = self.args.num_pano_views*self.args.num_pano_sweeps
                        rgb = rgb.permute(0,2,3,1).reshape(num_cam_views, self.args.batch_size, rgb.shape[-2], rgb.shape[-1], 3)
                        depth = depth.expand(-1,3,-1,-1).permute(0,2,3,1).reshape(num_cam_views, self.args.batch_size, depth.shape[-2], depth.shape[-1], 3)

                        # Save attention over instruction
                        if t == 0:
                            att_ims = []
                            belief_ims = [[] for n in range(self.args.batch_size)]
                            for n in range(seq.shape[0]):
                                instruction = self.dataloader.tokenizer.decode_sentence(seq[n]).split()
                                act_att = act_att_weights[n].cpu().numpy()
                                obs_att = obs_att_weights[n].cpu().numpy()
                                valid_act_att = act_att[:, :seq_lens[n]].transpose()
                                valid_obs_att = obs_att[:, :seq_lens[n]].transpose()
                                fig = viz_utils.plot_att_graph(valid_act_att, valid_obs_att, instruction, seq_lens[n].item(), black_background=True)
                                if self.args.viz_gif:
                                    att_ims.append(viz_utils.figure_to_rgb(fig))
                                else:
                                    fig.savefig("%s/attention-%s-it%d-n%d.png" % (self.args.viz_folder, split, it, n), facecolor=fig.get_facecolor(), transparent=True)
                                plt.close('all')

                        for k in range(3):
                            if self.args.policy_gt_belief:
                                viz = floor_maps
                            else:
                                viz = self.overlay_mask(floor_maps, mask)
                            if k>=1:
                                viz = self.overlay_belief(viz, belief_pred)
                            viz = self.overlay_goal(viz, self.dataloader.goal_coords()+self.mapper.map_center[:,:2])
                            if k>=2:
                                viz = self.overlay_local_graph(viz, features, action_idx)
                            else:
                                viz = self.overlay_local_graph(viz, features)
                            for n in range(len(viz)):
                                if not ended[n]:
                                    if self.args.viz_gif:
                                        image = viz[n]*255
                                        # Add attention image on left
                                        min_val = image.shape[0]//2-att_ims[n].shape[0]//2
                                        max_val = min_val + att_ims[n].shape[0]
                                        image = np.flip(image[min_val:max_val,min_val:max_val,:],2)
                                        image = np.concatenate([att_ims[n],image], axis=1)
                                        # Add rgb images at bottom
                                        new_width = int(image.shape[-2]/float(rgb.shape[0]))
                                        new_height = int(new_width*rgb.shape[-3]/float(rgb.shape[-2]))
                                        rgb_ims = [cv2.resize(rgb[i,n].cpu().detach().numpy(), (new_width,new_height)) for i in range(rgb.shape[0])]
                                        rgb_ims = np.concatenate(rgb_ims, axis=1)
                                        # Add depth images at bottom
                                        depth_ims = [cv2.resize(depth[i,n].cpu().detach().numpy()/200.0, (new_width,new_height)) for i in range(depth.shape[0])]
                                        depth_ims = np.concatenate(depth_ims, axis=1)
                                        image = np.concatenate([image,rgb_ims,depth_ims], axis=0)
                                        belief_ims[n].append(image.astype(np.uint8))
                                    else:
                                        filename = '%s/belief-%s-it%d-n%d-t%d_%d.png' % (self.args.viz_folder, split, it, n, t, viz_counter)
                                        cv2.imwrite(filename, viz[n]*255)
                            viz_counter += 1

                    ended |= sim.takeMultiStepAction(action_idx)

                    if ended.all():
                        break

                if self.args.viz_gif and self.args.viz_eval and it<self.args.viz_iterations:
                    import imageio
                    for n in range(self.args.batch_size):
                        filename = '%s/%s-%s-it%d-n%d' % (self.args.viz_folder, split, batch[n]['instr_id'], it, n)
                        if not os.path.exists(filename):
                            os.makedirs(filename)
                        with imageio.get_writer(filename+'.gif', mode='I', format='GIF-PIL', subrectangles=True, fps=1) as writer:
                            for i,image in enumerate(belief_ims[n]):
                                writer.append_data(image)
                                cv2.imwrite("%s/%04d.png" % (filename,i), np.flip(image,2))


        # Eval
        out_dir = "%s/%s" % (args.result_dir, args.exp_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        output_file = "%s/%s_%s_%d.json" % (out_dir, split, args.exp_name, args.val_epoch)
        with open(output_file, 'w') as f:
            json.dump(sim.traj, f)
        scores = vln_eval.score(output_file)
        print(scores)
        with open(output_file.replace('.json', '_scores.json'), 'w') as f:
            json.dump(scores, f)



if __name__ == "__main__":

    args = cfg.parse_args()
    args_text = cfg.print_args(args)

    if args.eval is False:
        trainer = PolicyTrainer(args=args)
        trainer.train()
    else:
        filepath = "%s/%s/%s_%d" % (args.snapshot_dir, args.exp_name, args.exp_name, args.val_epoch)
        evaluator = PolicyEvaluator(args=args, filepath=filepath)
        evaluator.evaluate("val_unseen")
        evaluator.evaluate("val_seen")
