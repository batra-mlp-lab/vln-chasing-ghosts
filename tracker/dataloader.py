import os
import random
import math
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

import text_utils as tu
from utils import load_nav_graphs, xyzhe_from_viewpoints


class DataLoader:
    """ Class to handle R2R data loading, instruction encoding and batch sampling """

    def __init__(self, args, splits=['train']):

        self.args = args

        # Check for vocabs, build if necessary
        if not os.path.exists(self.args.train_vocab):
            tu.write_vocab(
                tu.build_vocab(self.args.data_dir, splits=["train"]),
                self.args.train_vocab,
            )
        if not os.path.exists(self.args.trainval_vocab):
            tu.write_vocab(
                tu.build_vocab(self.args.data_dir, splits=["train", "val_seen", "val_unseen"]),
                self.args.trainval_vocab,
            )

        # Load vocab and set tokenizer etc
        if self.args.train_split == 'trainval':
            self.vocab = tu.read_vocab(self.args.trainval_vocab)
        else:
            self.vocab = tu.read_vocab(self.args.train_vocab)
        self.args.vocab_size = len(self.vocab)
        self.tokenizer = tu.Tokenizer(self.vocab, self.args.max_input_length)

        # Load training and val data
        # data: [{"scan", "instr_id", "path", "heading", "instruction",  "encoding", "xyzhe"}]
        self.data = []
        self.scans = []
        for item in tu.load_datasets(self.args.data_dir, splits):
            # Split multiple instructions into separate entries
            for j,instr in enumerate(item['instructions']):
                self.scans.append(item['scan'])
                train_obj = {
                    'heading': item['heading'],
                    'scan': item['scan'],
                    'instruction': instr,
                    'instr_id': '%d_%d' % (item['path_id'], j),
                    'encoding': self.text_to_encoding(instr),
                    'path': item['path']
                }
                self.data.append(train_obj)
        self.scans = set(self.scans)
        self.splits = splits
        random.shuffle(self.data)
        self.reset_epoch()
        print('Loaded %d data tuples from %s' % (len(self.data), self.splits))

        # Load connectivity graph for each scan, useful for reasoning about shortest paths
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

        # Pre-calculate the position and heading along each path
        for item in self.data:
            G = self.graphs[item['scan']]
            item['xyzhe'] = xyzhe_from_viewpoints(G, item['path'], item['heading'])


    def text_to_encoding(self, instr):
        """ Returns a fixed size numpy array of indices. """
        return self.tokenizer.encode_sentence(instr)


    def reset_epoch(self):
        """ Reset the data index to beginning of epoch. Primarily for testing. """
        self.ix = 0


    def _next_minibatch(self):
        """ Internal function to sample next random minibatch """
        batch = self.data[self.ix:self.ix+self.args.batch_size]
        if len(batch) < self.args.batch_size:
            random.shuffle(self.data)
            self.ix = self.args.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.args.batch_size
        self.batch = np.array(batch)


    def _sort_batch(self):
        """ Internal function to extract instructions from a list of observations
            and sort by descending sequence length (to enable PyTorch packing). """

        seq_tensor = np.array([item['encoding'] for item in self.batch])
        seq_lengths = np.argmax(seq_tensor == self.args.enc_padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

        seq_tensor = torch.from_numpy(seq_tensor).to(self.args.device)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == self.args.enc_padding_idx)[:,:seq_lengths[0]]

        if self.args.batch_size > 1:
            self.batch = self.batch[perm_idx.cpu()]

        return sorted_tensor.long(), mask.byte(), seq_lengths


    def get_batch(self):
        """
        Returns:
            seq (torch.cuda.LongTensor): Instruction encodings (sorted by decreasing length)
            seq_mask (torch.cuda.ByteTensor): Mask for invalid indices in seq
            seq_lens (torch.cuda.LongTensor): Seq lengths
            batch (np.array): [{"scan", "instr_id", "path", "heading", "instruction",  "encoding"}]

        Shape:
            seq: (batch_size, max_input_length)
            mask_mask: (batch_size, max_seq_length), max_seq_length is length of longest seq
            seq_lens: (batch_size)
        """
        self._next_minibatch()
        seq,seq_mask,seq_lens = self._sort_batch()
        return seq,seq_mask,seq_lens,self.batch


    # TODO refactor to use this all-inclusive function everywhere, and remove all the other overlapping functions below!
    # TODO anything requiring these coordinates in map space should be done by the mapper
    def path_xyzhe(self):
        """ Return (x,y,z,heading,elevation) of each path step in the current minibatch. The tensor xyzhe is padded
            using the last viewpoint in the path sequence, but the number of valid viewpoints is given by path_lens.
        Outputs:
            xyzhe: (K, N, 5),
            path_lens: (N)
        """
        xyzhe = torch.zeros(self.args.max_steps+1, self.args.batch_size, 5, device='cpu')
        path_lens = torch.zeros(self.args.batch_size, device='cpu')
        assert len(self.batch) == self.args.batch_size
        for n,item in enumerate(self.batch):
            num_views = item['xyzhe'].shape[0]
            path_lens[n] = num_views
            xyzhe[:num_views, n, :] = torch.from_numpy(item['xyzhe'])
            xyzhe[num_views:, n, :] = xyzhe[num_views-1, n, :] # Fill with the last value
        return xyzhe.to(self.args.device), path_lens.to(self.args.device)

    def convert_map_pixels_to_xy_coords(self, pixels, map_center, multi_timestep_input=False, downsample=1):
        """
            Converts input in map pixels to real world xy coordinates
            Input can be of shape (batch_size, 2) or (batch_size, timesteps, 2), set
            multi_timestep_input accordingly
        """
        # shape: (batch_size, 2)
        xy_offset = map_center[:, :2]
        out_size = torch.tensor([self.args.map_range_x, self.args.map_range_y], dtype=torch.float, device=self.args.device)
        gridcellsize = self.args.gridcellsize * downsample
        pixels = pixels.float()

        if multi_timestep_input:
            # shape: (batch_size, timesteps, 2)
            xy_offset = xy_offset.unsqueeze(1).expand(-1, self.args.timesteps, -1)
            out_size = out_size.unsqueeze(0).unsqueeze(0).expand(self.args.batch_size, self.args.timesteps, -1)

        pos_xy = (pixels - 0.5*out_size)*gridcellsize - xy_offset
        return pos_xy


    def get_goal_coords_on_map_grid(self, map_center):
        pos = self.goal_coords()
        out_size = torch.tensor([self.args.map_range_x, self.args.map_range_y], dtype=torch.float, device=self.args.device)
        xy_offset = map_center[:,:2]
        goal = ((pos + xy_offset)/self.args.gridcellsize + 0.5*out_size).round().long()
        return goal


    def goal_coords(self):
        """ Return the (x,y) coordinate of each goal location in the current minibatch
        Outputs:
            xy: (N, 2)
        """
        pos = [self.graphs[item['scan']].nodes[item['path'][-1]]["position"][:2] for item in self.batch]
        return torch.tensor(pos, device=self.args.device, dtype=torch.float)


    def path_coords(self):
        """ Return the (x,y) coordinate of each path step in the current minibatch
        Outputs:
            xy: (K, N, 2)
        """
        # Hack for ignore values - they will be too large to fall inside the map
        pos = 1000000*torch.ones(self.args.max_steps, self.args.batch_size, 2, device='cpu')
        for i,item in enumerate(self.batch):
            for k,vId in enumerate(item['path'][1:]):
                pos[k,i] = torch.from_numpy(self.graphs[item['scan']].nodes[vId]["position"][:2])
        return pos.to(self.args.device)


    def closest_to_goal(self, G):
        """ Return the distance to goal for provided .
            Ignore value is -1.
        Outputs:
            dist: (T, N, 1)
        """
        batch_indices = [int(vId.split('_')[0]) for vId in sorted(G.nodes)]
        targets = torch.empty(self.args.batch_size, device=self.args.device).long()
        dist = [float("inf") for n in range(self.args.batch_size)]
        for i,node in enumerate(sorted(G.nodes)):
            n,vId = node.split('_')
            n = int(n)
            goal_vId = self.batch[n]['path'][-1]
            new_dist = self.distances[self.batch[n]['scan']][vId][goal_vId]
            if new_dist < dist[n]:
                dist[n] = new_dist
                targets[n] = i - batch_indices.index(n)
        return targets


    def get_supervision(self, viewpointIds):
        """ Return a list of optimal next viewpointIds based on the current batch. """
        next = []
        for item,vid in zip(self.batch,viewpointIds):
            goalId = item['path'][-1]
            if vid == goalId:
                next.append(vid) # Your already there!
            else:
                path = self.paths[item['scan']][vid][goalId]
                next.append(path[1]) # Next step
        return np.array(next)


    def get_path(self, dist=1.0):
        """ Return a list of (x,y,prev_x,prev_y) ground truth paths for each item in the batch,
            to support visualization and debugging """
        paths = []
        for i,item in enumerate(self.batch):
            paths.append([])
            for j,vid in enumerate(item['path']):
                node = self.graphs[item['scan']].nodes[vid]
                x,y = node["position"][:2]
                if j == 0:
                    heading = 0.5 * math.pi - item["heading"]
                    front_x = x + dist * math.cos(heading)
                    front_y = y + dist * math.sin(heading)
                else:
                    dist = math.sqrt((x-prev_x)**2 + (y-prev_y)**2)
                    front_x = x + dist*(x-prev_x)/dist
                    front_y = y + dist*(y-prev_y)/dist
                paths[-1].append((x,y,front_x,front_y))
                prev_x = x
                prev_y = y
        return paths
