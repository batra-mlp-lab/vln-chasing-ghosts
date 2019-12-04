import math
import random
import numpy as np
import torch
import utils
import time
import networkx as nx
import scipy.sparse as sp

import MatterSim


class PanoSimulator(MatterSim.Simulator):
    """ Simulator that can provide panoramic environment views. """

    def __init__(self, args):
        super(PanoSimulator, self).__init__()
        self.args = args
        if self.args.preloading:
            print("Simulator images will be preloaded")
        prev_time = time.time()
        self.setPreloadingEnabled(self.args.preloading)
        self.setDepthEnabled(True)
        self.setRestrictedNavigation(False)
        self.setBatchSize(self.args.batch_size)
        self.setCacheSize(self.args.cache_size)
        self.setCameraResolution(self.args.image_width, self.args.image_height)
        self.setCameraVFOV(self.args.vfov)
        self.initialize()
        self._reset_visited_goal()
        prev_time, time_taken = utils.time_it(prev_time)
        print("Simulator initialized: %0.2f secs" % time_taken)

    def _reset_visited_goal(self):
        self.visited_goal = np.array([False for i in range(self.args.batch_size)])

    def newEpisode(self, batch):
        """ New episode and set elevation to default, using data struct from DataLoader """
        self._reset_visited_goal()
        scanIds = [item['scan'] for item in batch]
        viewpointIds = [item['path'][0] for item in batch]
        headings = [item['heading'] for item in batch]
        super(PanoSimulator, self).newEpisode(scanIds, viewpointIds, headings,
              [self.args.elevation] * self.args.batch_size)

    def newRandomEpisode(self, scanIds):
        """ Random episode and adjust elevation (initially zero) to default """
        self._reset_visited_goal()
        super(PanoSimulator, self).newRandomEpisode(scanIds)
        self.makeAction([0]*self.args.batch_size, [0]*self.args.batch_size,
              [self.args.elevation]*self.args.batch_size)

    def takeRandomAction(self):
        """Move randomly to neighbouring viewpoint
        """
        states = self.getState()
        next_location = []
        for n, state in enumerate(states):
            visitable_viewpoints = len(state.navigableLocations) - 1
            if visitable_viewpoints == 0:
                idx = 0
            else:
                idx = random.randint(1, visitable_viewpoints)
            next_location.append(idx)
        self.makeAction(next_location, [0]*self.args.batch_size, [0]*self.args.batch_size)



    def takePseudoSupervisedAction(self, supervision_function):
        """ Takes action based on the supervision function provided. Probabilty for taking the
        supervised action is args.supervision_prob and for rest, it is (1-supervision_prob)/(len-1).
        If no other navigable location is available, then don't move.
        """
        states = self.getState()
        curr_viewpoints = [state.location.viewpointId for state in states]
        target_viewpoints = supervision_function(curr_viewpoints)
        self.visited_goal |= np.array([curr_viewpoints[n]==target_viewpoints[n] for n in range(len(target_viewpoints))])
        next_location = []
        for n, state in enumerate(states):
            visitable_viewpoints = [location.viewpointId for location in state.navigableLocations]
            target_viewpoint_idx = visitable_viewpoints.index(target_viewpoints[n])
            l = len(visitable_viewpoints)
            if l == 1:
                idx = 0
            elif self.visited_goal[n]:
                idx = random.randint(1, l-1)
            else:
                choice = [i for i in range(l)]
                weights = [float(1-self.args.supervision_prob)/(l-1) for i in range(l)]
                weights[target_viewpoint_idx] = self.args.supervision_prob
                idx = np.random.choice(choice, size=1, p=weights)
            next_location.append(idx)
        self.makeAction(next_location, [0]*self.args.batch_size, [0]*self.args.batch_size)

    def takeActionBasedOnViewpointIds(self, target_viewpoints):
        """ (Figures out and) Takes action to go to the targetr viewpoint. Used for DFS """
        states = self.getState()
        next_location = []
        for n, state in enumerate(states):
            visitable_viewpoints = [location.viewpointId for location in state.navigableLocations]
            target_viewpoint_idx = visitable_viewpoints.index(target_viewpoints[n])
            next_location.append(target_viewpoint_idx)
        self.takeTurningAction(next_location)

    def takeTurningAction(self, index):
        """ Action method that will turn to face the viewpoint in the index while moving to it.
            Heading change will be zero for index==0, i.e. agent staying at same viewpoint """
        heading_chg = [state.navigableLocations[index[i]].rel_heading for i,state in enumerate(self.getState())]
        elevation_chg = [0]*self.args.batch_size
        self.makeAction(index, heading_chg, elevation_chg)

    def getImages(self):
        """ Return batched RGB and depth tensors with shape (N, C, H, W) """
        states = self.getState()
        rgb_array = np.array([np.array(state.rgb, copy=False, dtype=np.uint8) for state in states])
        depth_array = np.array([np.array(state.depth, copy=False, dtype=np.int32) for state in states])
        rgb = torch.from_numpy(np.transpose(rgb_array, (0, 3, 1, 2))).float().to(self.args.device)
        rgb = rgb.flip((1))  # convert BGR to RGB
        depth = torch.from_numpy(np.transpose(depth_array, (0, 3, 1, 2))).float().to(self.args.device)
        return rgb,depth

    def getPanos(self):
        """ Return batched RGB and depth tensors with shape (N*num_pano_views*num_pano_sweeps, C, H, W) that provide
            views of the entire panorama, and the agent's state containing x, y, z, heading, elevation
            (N*num_pano_views, 5) """
        num_views = self.args.num_pano_views
        num_sweeps = self.args.num_pano_sweeps
        N = self.args.batch_size

        # Turn to face heading 0
        start_heading = [-state.heading for state in self.getState()]
        super(PanoSimulator, self).makeAction([0]*N, start_heading, [0]*N)
        for sweep in range(num_sweeps):
            for view in range(num_views):
                states = self.getState()
                rgb_array = np.array([np.array(state.rgb, copy=False, dtype=np.uint8) for state in states])
                depth_array = np.array([np.array(state.depth, copy=False, dtype=np.int32) for state in states])
                if view == 0 and sweep == 0:
                    h = rgb_array.shape[1]
                    w = rgb_array.shape[2]
                    rgb = torch.empty(N*num_views*num_sweeps, 3, h, w, device=self.args.device)
                    depth = torch.empty(N*num_views*num_sweeps, 1, h, w, device=self.args.device)
                    S = torch.empty(N*num_views*num_sweeps, 5, device=self.args.device)

                ix = sweep*num_views+view
                rgb[ix*N:(ix+1)*N, :, :, :] = torch.from_numpy(np.transpose(rgb_array, (0, 3, 1, 2)))
                depth[ix*N:(ix+1)*N, :, :, :] = torch.from_numpy(np.transpose(depth_array, (0, 3, 1, 2)))
                S[ix*N:(ix+1)*N, :] = torch.Tensor([[s.location.x, s.location.y, s.location.z,
                                          s.heading, s.elevation] for s in states]).to(self.args.device)

                # Move the sim viewpoint so it ends in the same place
                elev = 0
                heading_chg = [math.pi*2/num_views]*N
                if view+1==num_views: # Last viewpoint in sweep
                    if sweep+1 < num_sweeps: # Another sweep to come, so elevate up
                        elev = self.args.elevation_chg
                    else: # last viewpoint in last sweep, so reset to start
                        elev = -self.args.elevation_chg*(num_sweeps-1)
                        heading_chg = [-start_heading[i]-state.heading for i,state in enumerate(self.getState())]
                super(PanoSimulator, self).makeAction([0]*N, heading_chg, [elev]*N)

        rgb = rgb.flip((1))  # convert BGR to RGB
        return rgb,depth,S

    def getXYZHE(self):
        xyzhe = [[s.location.x, s.location.y, s.location.z, s.heading, s.elevation] for s in self.getState()]
        return torch.Tensor(xyzhe).to(self.args.device)

    def getNavLocations(self):
        """ Return robot-relative heading,elevation and distance to navigable locations, plus the associated viewpoint
            ids and a matchin mask. The robots current location (the stay-put option) is at index 0.
        Shape:
            Output:
                hed: (N, max_graph_degree+1, 3): robot relative heading, elevation and distance
                xyzhe: (N, max_graph_degree+1, 5): absolute x,y,z,heading,elevation
                mask: (N, max_graph_degree+1) : valid == 0, invalid == 1
                viewpointIds: (N, max_graph_degree+1)
        """
        hed = torch.zeros(self.args.batch_size, self.args.max_graph_degree+1, 3, device=self.args.device)
        xyzhe = torch.zeros(self.args.batch_size, self.args.max_graph_degree+1, 5, device=self.args.device)
        viewpointIds = []
        for n,state in enumerate(self.getState()):
            data = [[loc.rel_heading, loc.rel_elevation, loc.rel_distance] for loc in state.navigableLocations]
            hed[n,:len(data)] = torch.tensor(data, device=self.args.device)
            abs_data = [[loc.x, loc.y, loc.z, loc.rel_heading + state.heading, state.elevation] for loc in state.navigableLocations]
            xyzhe[n,:len(abs_data)] = torch.tensor(abs_data, device=self.args.device)
            vpts = [l.viewpointId for l in state.navigableLocations]
            vpts += [[None] for i in range(self.args.max_graph_degree+1-len(data))]
            viewpointIds.append(vpts)
        mask = hed[:,:,2] == 0
        mask[:,0] = 0 # Stay put option will have distance 0 but is still valid
        return hed,xyzhe,mask,np.array(viewpointIds)


class PanoSimulatorWithGraph(PanoSimulator):
    """ PanoSimulator that stores observed pano locations into a navigation graph, so the agent
        can make global decisions to move to any previously observed location. """

    def __init__(self, args, disable_rendering=False):
        if disable_rendering:
            self.setRenderingEnabled(False)
        super(PanoSimulatorWithGraph, self).__init__(args)
        self.record = False

    def record_traj(self, value):
        self.record = value
        if self.record:
            self.traj = []

    def node_str(self, n, viewpointId):
        return '%d_%s' % (n, viewpointId)

    def _add_trajectories(self, batch):
        if self.record:
            for n,state in enumerate(self.getState()):
                self.traj.append({
                    'instr_id': batch[n]['instr_id'],
                    'trajectory': [(state.location.viewpointId, state.heading, state.elevation)]
                })

    def _record_visible_navgraph(self):
        for n,state in enumerate(self.getState()):
            curr_node = self.node_str(n, state.location.viewpointId)
            for k,loc in enumerate(state.navigableLocations):
                loc_node = self.node_str(n, loc.viewpointId)
                if loc_node != curr_node:
                    self.G.add_edge(curr_node, loc_node, weight=loc.rel_distance)
            for k,loc in enumerate(state.navigableLocations):
                loc_node = self.node_str(n, loc.viewpointId)
                self.G.node[loc_node]['position'] = torch.tensor([loc.x, loc.y, loc.z], device=self.args.device)
            self.G.node[curr_node]['visited'] = True

    def newEpisode(self, batch):
        """ New episode and set elevation to default, using data struct from DataLoader """
        super(PanoSimulatorWithGraph, self).newEpisode(batch)
        self.start_node = [self.node_str(n,state.location.viewpointId) for n,state in enumerate(self.getState())]
        self.G = nx.Graph()
        self._record_visible_navgraph()
        self._add_trajectories(batch)

    def newRandomEpisode(self, batch):
        """ Random episode and adjust elevation (initially zero) to default """
        scanIds = [item['scan'] for item in batch]
        super(PanoSimulatorWithGraph, self).newRandomEpisode(scanIds)
        self.G = nx.Graph()
        self._record_visible_navgraph()
        self._add_trajectories(batch)

    def makeAction(self, index, heading, elevation):
        super(PanoSimulatorWithGraph, self).makeAction(index, heading, elevation)
        self._record_visible_navgraph()
        if self.record:
            for n,state in enumerate(self.getState()):
                item = (state.location.viewpointId, state.heading, state.elevation)
                if self.traj[n-self.args.batch_size]['trajectory'][-1] != item:
                    self.traj[n-self.args.batch_size]['trajectory'].append(item)

    def _compute_shortest_paths(self):
        self.distances = dict(nx.all_pairs_dijkstra_path_length(self.G))
        self.paths = dict(nx.all_pairs_dijkstra_path(self.G))

    def getGraphNodes(self):
        """ Return graph with xyz coords in the simulator, distance from the start,
            distance from the agent's current location, and whether the node has been visited.
            Output:
                features: (K, 7): batch_index, rel_x, rel_y, rel_z,
                              distance_from_start,distance_from_loc,visited
                adj: adjacency matrix
                shortest_paths: (K, max_steps, 3): xyz for the shortest path from start to each node
        """
        self._compute_shortest_paths()
        curr_node = [self.node_str(n,state.location.viewpointId) for n,state in enumerate(self.getState())]
        num_nodes = nx.number_of_nodes(self.G)
        features = torch.empty(num_nodes, 7, device=self.args.device)
        i = 0
        shortest_paths = torch.zeros(num_nodes, self.args.max_steps*2, 3, device=self.args.device)
        for node in sorted(self.G.nodes(data=True)):
            n = int(node[0].split('_')[0])
            start_pos = self.G.node[self.start_node[n]]['position']
            features[i,0] = n
            features[i,1:4] = node[1]['position'] - start_pos # relative xyz (relative to start location)
            features[i,4] = self.distances[self.start_node[n]][node[0]] # dist from start

            path = self.paths[self.start_node[n]][node[0]] # path from start to node
            shortest_paths[i,:len(path),:] = torch.cat([self.G.node[p]['position'].unsqueeze(0) for p in path], dim=0)

            features[i,5] = self.distances[curr_node[n]][node[0]] # dist from curr
            features[i,6] = 1 if 'visited' in node[1] else 0
            i = i+1
        # Calculate adjacency matrix
        adj = nx.adjacency_matrix(self.G,nodelist=sorted(self.G.nodes()))
        adj[adj>0]=1
        adj = sp.eye(adj.shape[0])
        adj = self._sparse_mx_to_torch_sparse_tensor(adj).to(self.args.device)
        return features,adj,shortest_paths

    def takeMultiStepAction(self, action_idx):
        """ Takes a series of actions to move to the nominated graph node. The sim stops prematurely
            if it encounters a node that has not previously been visited (so the mapper can be updated).
            The episode is entered if the nominated node has previously been visited.
        """
        # Find all shortest paths
        self._compute_shortest_paths()
        curr_vId = [self.node_str(n,state.location.viewpointId) for n,state in enumerate(self.getState())]

        # Map actions to shortest paths in the known graph
        paths = []
        max_len = 0
        prev_n = -1
        for i,vId in enumerate(sorted(self.G.nodes())):
            n = int(vId.split('_')[0])
            if n>prev_n:
                start_ix = i
            ix = i - start_ix
            if ix == action_idx[n]:
                paths.append(self.paths[curr_vId[n]][vId])
                max_len = max(max_len,len(paths[-1]))
            prev_n = n

        # Step through paths
        ended = torch.zeros(self.args.batch_size, device=self.args.device)
        for step in range(1, max(2,max_len)):
            targets = []
            for n,path in enumerate(paths):
                if step < len(path):
                    # Move to next step in path
                    vId = path[step]
                    if 'visited' in self.G.node[vId]:
                        if step+1 == len(path):
                            # nominated node is previously visited, so we will stop
                            ended[n] = 1
                        else:
                            # encounters node not visited, delete rest of path
                            del path[step+1:]
                else:
                    # Stay in same place (filler action)
                    vId = path[-1]
                    if step == 1:
                        ended[n] = 1
                targets.append(vId.split('_')[1])
            self.takeActionBasedOnViewpointIds(targets)
            self._record_visible_navgraph()
        return ended.byte()


    def _normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
