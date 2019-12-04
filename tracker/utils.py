import os
import math
import numpy as np
import torch
import json
import random
import time
import datetime
import networkx as nx
import cv2


def load_scenes(split, splits_root="tracker/splits", connectivity_root="connectivity"):
    ''' Load a mapping from scanId to viewpoint count for a given dataset split (train/val/test) '''

    scenes_file_path = os.path.join(splits_root, "scenes_" + split + ".txt" )
    scenes_freq_map = {}
    with open(scenes_file_path) as f:
        for line in f:
            scene = line.strip()
            connectivity_json_path = os.path.join(connectivity_root, scene + "_connectivity.json")
            with open(connectivity_json_path) as json_data:
                viewpoints = json.load(json_data)
                no_of_viewpoints = sum([1 if viewpoint['included']==True else 0 for viewpoint in viewpoints])
                scenes_freq_map[scene] = no_of_viewpoints
    return scenes_freq_map


def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def xyzhe_from_viewpoints(G, viewpointIds, sim_start_heading):
    ''' Return xyz,heading,elevation for each point in the path. Heading is defined in
        according to the simulator convention, i.e. clockwise rotation in the xy plane
        from the positive y direction.  '''
    xyzhe = np.zeros((len(viewpointIds),5))
    for i,vId in enumerate(viewpointIds):
        xyz = G.nodes[vId]["position"][:3]
        xyzhe[i,:3] = xyz
        if i == 0:
            xyzhe[i,3] = sim_start_heading
        else:
            xyzhe[i,3] = 0.5 * math.pi - math.atan2(xyz[1]-xyz_prev[1], xyz[0]-xyz_prev[0])
        xyz_prev = xyz
    return xyzhe


def compute_argmax(ten):
    """Compute argmax for 2D grid for tensors of shape (batch_size, size_y, size_x)

    Args:
        ten (torch.[cuda].FloatTensor): (batch_size, size_y, size_x)
    Returns:
        indices (torch.[cuda].LongTensor): (batch_size, 2) index order: (y, x)
    """

    batch_size = ten.shape[0]
    size_y = ten.shape[1]
    size_x = ten.shape[2]

    # shape: flattened_indices (batch_size)
    flattened_indices = torch.argmax(ten.view(batch_size, -1), dim=1)

    # shape: index_y (batch_size)
    # shape: index_x (batch_size)
    index_y = (flattened_indices // size_x)
    index_x = (flattened_indices % size_x)

    # shape: index_y (batch_size, 2)
    indices = torch.cat([index_y.unsqueeze(1), index_x.unsqueeze(1)], dim=1)

    return indices


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def time_it(prev_time):
    current_time = time.time()
    time_taken = datetime.timedelta(
        seconds=current_time - prev_time).total_seconds()
    # print("Time: %0.2f secs %s" % (time_taken, string) )
    return current_time, time_taken


def euclidean_distance(point1, point2):
    """Calculate batch average euclidean distance between 2 points (x,y)

    Args:
        point1 (torch[.cuda].FloatTensor): shape (batch_size, 2)
        point2 (torch[.cuda].FloatTensor): shape (batch_size, 2)
    """

    x = point2[:,0] - point1[:, 0]
    y = point2[:,1] - point1[:, 1]
    dist = (x**2 + y**2)**0.5
    return dist


def get_gaussian_blurred_map_mask(map_range_y, map_range_x, radius, blur_kernel):
    """ Compute a mask with a circle drawn at the radius pixels from centre, followed by gaussian
        blur.

    Args:
        map_range_y (int): Height of the mask
        map_range_x (int): Width of the mask
        radius (int): Radius at which the circle is to be drawn
        blur_kernel (tuple): (kernel_size_x, kernel_size_y), should be positive and odd

    Returns:
        (torch.cuda.FloatTensor): shape (map_range_y, map_range_x)
    """
    mask = np.zeros((map_range_y, map_range_x))
    cv2.circle(mask, (map_range_x//2, map_range_y//2), radius, 1)
    mask = cv2.GaussianBlur(mask, blur_kernel, 0)
    return torch.tensor(mask, dtype=torch.float, device='cuda')



def get_floorplan_images(states, floorplan, map_range_x, map_range_y, scale_factor=1):
    """ Get floorplan images from floorplan class """
    size_x = int(map_range_x * scale_factor)
    size_y = int(map_range_y * scale_factor)
    ims = floorplan.rgb(states, (size_x, size_y))
    return ims
