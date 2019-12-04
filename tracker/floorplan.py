
from collections import defaultdict
import numpy as np
import cv2
import math
import csv


FLOORPLAN_DIR = "data/v1/scans/floorplans/"
RGB_TEMPLATE = "out_dir_rgb_png/output_%s_level_%d.0.png"
SEMANTIC_TEMPLATE = "out_dir_semantic_png/output_%s_level_%d.0.png"
SEMANTIC_OUTPUT_TEMPLATE = "out_dir_semantic_png/gray_output_%s_level_%d.0.png"
RENDER_CONFIG = "render_config.csv"
RENDER_BUFFER = 0.5 # meters added to x,y bbox coordinates during rendering

RGB_to_mcat40 = {
  (0, 0, 0): 0,
  (0, 0, 255): 1,
  (0, 255, 0): 2,
  (0, 255, 255): 3,
  (255, 0, 255): 4,
  (255, 128, 0): 5,
  (0, 255, 128): 6,
  (128, 0, 255): 7,
  (128, 255, 0): 8,
  (0, 128, 255): 9,
  (255, 0, 128): 10,
  (128, 0, 0): 11,
  (0, 128, 0): 12,
  (0, 0, 128): 13,
  (128, 128, 0): 14,
  (0, 128, 128): 15,
  (128, 0, 128): 16,
  (178, 0, 0): 17,
  (0, 178, 0): 18,
  (0, 0, 178): 19,
  (178, 178, 0): 20,
  (0, 178, 178): 21,
  (178, 0, 178): 22,
  (178, 76, 0): 23,
  (0, 178, 76): 24,
  (76, 0, 178): 25,
  (76, 178, 0): 26,
  (0, 76, 178): 27,
  (178, 0, 76): 28,
  (76, 0, 0): 29,
  (0, 76, 0): 30,
  (0, 0, 76): 31,
  (76, 76, 0): 32,
  (0, 76, 76): 33,
  (76, 0, 76): 34,
  (255, 76, 76): 35,
  (76, 255, 76): 36,
  (76, 76, 255): 37,
  (255, 255, 76): 38,
  (76, 255, 255): 39,
  (255, 76, 255): 40
}

mcat40_labels = [
    'void',             # 0
    'wall',             # 1
    'floor',            # 2
    'chair',            # 3
    'door',             # 4
    'table',            # 5
    'picture',          # 6
    'cabinet',          # 7
    'cushion',          # 8
    'window',           # 9
    'sofa',             # 10
    'bed',              # 11
    'curtain',          # 12
    'chest_of_drawers', # 13
    'plant',            # 14
    'sink',             # 15
    'stairs',           # 16
    'ceiling',          # 17
    'toilet',           # 18
    'stool',            # 19
    'towel',            # 20
    'mirror',           # 21
    'tv_monitor',       # 22
    'shower',           # 23
    'column',           # 24
    'bathtub',          # 25
    'counter',          # 26
    'fireplace',        # 27
    'lighting',         # 28
    'beam',             # 29
    'railing',          # 30
    'shelving',         # 31
    'blinds',           # 32
    'gym_equipment',    # 33
    'seating',          # 34
    'board_panel',      # 35
    'furniture',        # 36
    'appliances',       # 37
    'clothes',          # 38
    'objects',          # 39
    'misc',             # 40
]

mcat40_to_RGB = {s_class: pixel for pixel, s_class in RGB_to_mcat40.items()}

occupied_class_to_class = {
    -1: 2,      # floor class to be ignored
    0:  [0,1,4,6,9,12,17,20,21,24,28,29,32,35,36,38,39,40],    # void+everything_else
    1:  3,     # chair
    2:  5,     # table
    3:  7,     # cabinet
    4:  8,     # cushion
    5:  10,    # sofa
    6:  11,    # bed
    7:  13,    # chest_of_drawers
    8:  14,    # plant
    9:  15,    # sink
    10: 16,    # stairs
    11: 18,    # toilet
    12: 19,    # stool
    13: 22,    # tv_monitor
    14: 23,    # shower
    15: 25,    # bathtub
    16: 26,    # counter
    17: 27,    # fireplace
    18: 30,    # railing
    19: 31,    # shelving
    20: 33,    # gym_equipment
    21: 34,    # seating
    22: 37,    # appliances
}

class_to_occupied_class = {}
for k,v in occupied_class_to_class.items():
    if(type(v)==list):
        for i in v:
            class_to_occupied_class[i] = k
    else:
        class_to_occupied_class[v] = k


class Floorplan:
    '''
    Outputs top-down RGB and semantic renderings of Matterport environments.
    NOTE: The world coordinates in the returned image are x-right, y-up, z-out, i.e. the
    world coordinates origin is at the bottom left, and the y-axis is flipped.
    '''

    def __init__(self, render_pixels_per_meter, scanIds):
        '''
        Args:
            render_pixels_per_meter: scale factor determining the required output resolution
            scanIds: list of Matterport environments (which will be pre-loaded)
        '''
        self.ppm = float(render_pixels_per_meter)
        self.rgb_fps = defaultdict(list)
        self.semantic_fps = defaultdict(list)
        # Preload and scale floorplan renders. The renders have a bottom-left origin (y-up, x-right)
        for scanId in scanIds:
            # print('Loading floorplan: %s' % scanId)
            level = 0
            while True:
                bgr = cv2.imread(FLOORPLAN_DIR + RGB_TEMPLATE % (scanId,level))
                if bgr is None:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.rgb_fps[scanId].append(rgb)
                class_render = cv2.imread(FLOORPLAN_DIR + SEMANTIC_OUTPUT_TEMPLATE % (scanId,level), cv2.IMREAD_GRAYSCALE)
                self.semantic_fps[scanId].append(class_render)
                level += 1
        # Preload render config (contains bounding boxes and other info for each floorplan)
        self.render_configs = defaultdict(list)
        with open(FLOORPLAN_DIR + RENDER_CONFIG) as csvfile:
            reader = csv.DictReader(csvfile)
            # fieldnames = ['scanId', 'level', 'x_low', 'y_low', 'z_low', 'x_high', 'y_high', 'z_high', 'width', 'height']
            for item in reader:
                self.render_configs[item['scanId']].append(item)


    def get_floorplan_transform_(self, scanId, xyz, heading, output_size):
        '''
        Determine which level a viewpoint is on, and the correct 2D affine transformation
        to apply to the associated floorplan.
        '''
        z_values = np.array([float(item['z_high']) for item in self.render_configs[scanId]])
        level_idx = np.abs(z_values - xyz[2]).argmin() # Select level
        config = self.render_configs[scanId][level_idx]
        # Determine the pixels per meter of the pre-rendered floorplan
        render_ppm_x = float(config['width']) / (float(config['x_high']) - float(config['x_low']) + 2.0 * RENDER_BUFFER)
        render_ppm_y = float(config['height']) / (float(config['y_high']) - float(config['y_low']) + 2.0 * RENDER_BUFFER)
        scale_x = self.ppm / render_ppm_x
        scale_y = self.ppm / render_ppm_y
        # Find the image bbox in world space (x-min, y-min)
        origin_x = float(config['x_low'])-RENDER_BUFFER
        origin_y = float(config['y_high'])+RENDER_BUFFER - float(output_size[1])/(self.ppm)
        # Redefine xyz to be at the bottom left of the rendered image
        xyz = (xyz[0] - float(output_size[0])/(2*self.ppm), xyz[1] - float(output_size[1])/(2*self.ppm), xyz[2])
        # Now translate the image so the corner of this bbox is at xyz
        tx = - (xyz[0] - origin_x) * self.ppm
        ty =   (xyz[1] - origin_y) * self.ppm
        M = np.array([[scale_x,        0, tx],
                      [      0,  scale_y, ty],
                      [      0,        0,  1]])
        R = cv2.getRotationMatrix2D((output_size[0]/2,output_size[1]/2), math.degrees(heading), 1)
        R = np.concatenate([R,np.array([[0,0,1]])], axis=0)
        M = np.matmul(R,M)[:-1]
        return level_idx,config,M


    def rgb(self, states, output_size, rotate=False):
        '''
        Output: a list of 3-channel RGB top-down floorplan images with the agent centered
        Args:
            states: input list of Matterport environment states
            output_size: (width, height) tuple for the returned floorplan images
            rotate: when true the floorplan is rotated to match the agent's heading.
                    As in the simulator, heading is defined from the y-axis with the
                    z-axis up (turning right is positive).
        '''
        renders = []
        for state in states:
            h = state.heading if rotate else 0
            xyz = (state.location.x, state.location.y, state.location.z)
            level_idx, config, M = self.get_floorplan_transform_(state.scanId, xyz, h, output_size)
            floorplan = self.rgb_fps[state.scanId][level_idx]
            output = cv2.warpAffine(floorplan, M, output_size, flags=cv2.INTER_AREA)
            renders.append(output)
        return renders


    def semantic(self, states, output_size, rotate=False):
        '''
        Output: a list of single-channel (grayscale) semantic top-down floorplan images with
                the agent centered. Each intensity value corresponds to a semantic class
                according to the mcat40 labels.
        Args:
            states: input list of Matterport environment states
            output_size: (width, height) tuple for the returned floorplan images
            rotate: when true the floorplan is rotated to match the agent's heading.
                    As in the simulator, heading is defined from the y-axis with the
                    z-axis up (turning right is positive).
        '''
        renders = []
        for state in states:
            h = state.heading if rotate else 0
            xyz = (state.location.x, state.location.y, state.location.z)
            level_idx, config, M = self.get_floorplan_transform_(state.scanId, xyz, h, output_size)
            floorplan = self.semantic_fps[state.scanId][level_idx]
            output = cv2.warpAffine(floorplan, M, output_size, flags=cv2.INTER_NEAREST)
            renders.append(output)
        return renders



def convert_semantic_map(scanId):
    '''
    Legacy code that was used to convert RGB semantic renders (generated using the mpview software
    from https://github.com/niessner/Matterport) to grayscale images such that each intensity value
    corresponds to a semantic class based on mcat40 labels.
    '''
    print('Loading floorplans: %s' % scanId)
    level = 0
    while True:
        bgr = cv2.imread(FLOORPLAN_DIR + SEMANTIC_TEMPLATE % (scanId,level))
        if bgr is None:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        class_render = np.apply_along_axis(rgb_to_class, 2, rgb)
        print(class_render.shape)
        assert cv2.imwrite(FLOORPLAN_DIR + SEMANTIC_OUTPUT_TEMPLATE % (scanId,level), class_render)
        level += 1


def rgb_to_class(pixel):
    ''' Map RGB pixel to semantic class '''
    rgb = tuple(pixel)
    if rgb in RGB_to_mcat40:
        return RGB_to_mcat40[rgb]
    else:
        return 0

def class_to_rgb(s_class):
    ''' Map semantic class to RGB pixel'''
    if s_class in mcat40_to_RGB:
        return mcat40_to_RGB[s_class]
    else:
        raise ValueError('Semantic class not in bounds: %d' % s_class)

def return_occupied_class_to_class(o_class):
    ''' Map occupied semantic class to RGB pixel'''
    if(o_class==0):
        s_class = 0
    else:
        if o_class in occupied_class_to_class:
            s_class = occupied_class_to_class[o_class]
        else:
            raise ValueError('Occupied semantic class not in bounds')
    return class_to_rgb(s_class)

def return_class_to_occupied_class(s_class):
    ''' Map semantic class to occupied semantic class'''
    if s_class in class_to_occupied_class:
        return class_to_occupied_class[s_class]
    else:
        raise ValueError('Occupied semantic class not in bounds')


def return_class_to_rgb_labels():
    return mcat40_to_RGB, mcat40_labels


if __name__ == "__main__":
    # Convert RGB semantic renders to grayscale
    from utils import load_scenes
    from multiprocessing import Pool
    p = Pool(20)

    for split in ['train', 'val', 'test']:
        scanIds = load_scenes(split).keys()
        p.map(convert_semantic_map, scanIds)
