import os
import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def get_masks(floorplan_images, mask, goal_pred):
    masks = []
    goal_pred = (255*torch.flip(goal_pred, [1])).type(torch.ByteTensor).cpu()
    for i,fp in enumerate(floorplan_images):
        fp = cv2.cvtColor(fp,cv2.COLOR_RGB2BGR)
        im = 200*np.array(mask[i].squeeze().cpu())
        # Flip y-axis to match floorplan renders, so y axis is now up
        im = cv2.flip(im, 0)
        im = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        im = 0.3*fp + 0.7*im
        radius = im.shape[0]/128
        cv2.circle(im,(im.shape[1]/2, im.shape[0]/2), radius, (0,0,255), -1)
        max_goal_pos = np.unravel_index(goal_pred[i].argmax(), goal_pred[i].shape)
        cv2.circle(im,(max_goal_pos[1],max_goal_pos[0]), radius, (0,165,255), -1) # goal argmax
        masks.append(im)
    return masks


def get_floorplan_with_goal_path_maps(floorplan_images, goal_map, path_map, scale_factor=1, target=False):
    images = []
    goal_map = (255*torch.flip(goal_map, [1])).type(torch.ByteTensor).cpu()
    if path_map is not None:
        path_map = (255*torch.flip(path_map, [1])).type(torch.ByteTensor).cpu()
    for i,im in enumerate(floorplan_images):
        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        if target is False:
            im = im * 0.3
        if path_map is not None:
            im[:,:,0] = np.maximum(im[:,:,0], path_map[i]) # Show path in blue
        im[:,:,1] = np.maximum(im[:,:,1], goal_map[i]) # Show goal location in green
        radius = im.shape[0]/128
        max_goal_pos = np.unravel_index(goal_map[i].argmax(), goal_map[i].shape)
        cv2.circle(im,(im.shape[1]/2, im.shape[0]/2), radius, (0,0,255), -1) # center
        cv2.circle(im,(max_goal_pos[1],max_goal_pos[0]), radius, (0,165,255), -1) # goal argmax
        images.append(im)
    return images


def save_floorplans_with_maps(list_of_tensors, text, save_folder="viz"):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    total_items = len(list_of_tensors[0])
    for i in range(total_items):
        concate_images = [ims[i] for ims in list_of_tensors]
        im = np.concatenate(concate_images, axis=1)
        cv2.imwrite('%s/floorplan_%s_%d.png' % (save_folder, text, i), im)
    print("Saved floorplan: %s/floorplan_%s_i.png" % (save_folder, text))


def save_floorplans_with_belief_maps(list_of_tensors, list_belief_maps, max_steps, batch_size, timesteps, split, it, save_folder="viz"):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    total_items = len(list_of_tensors[0])
    for i in range(total_items):
        concate_images = [ims[i] for ims in list_of_tensors]
        im = np.concatenate(concate_images, axis=1)
        for k in range(max_steps):
            belief_maps = []
            for ims in list_belief_maps:
                b_map = cv2.cvtColor(ims[i][k], cv2.COLOR_GRAY2BGR)
                max_goal_pos = np.unravel_index(ims[i][k].argmax(), ims[i][k].shape)
                radius = b_map.shape[0]/128
                cv2.circle(b_map,(b_map.shape[1]/2, b_map.shape[0]/2), radius, (0,0,255), -1) # center
                cv2.circle(b_map,(max_goal_pos[1],max_goal_pos[0]), radius, (0,165,255), -1) # goal argmax
                belief_maps.append(b_map)
            im1 = np.concatenate(belief_maps, axis=1)
            final_im = np.concatenate([im, im1], axis=1)

            n = i/timesteps
            t = i%timesteps
            filename = '%s/belief-%s-it%d-n%d-t%d-k%d.png' % (save_folder, split, it, n, t, k)
            cv2.imwrite(filename, final_im)

    print("Saved floorplan: %s/%s-it%d.png" % (save_folder, split, it))


def save_floorplans(states, floorplan, map_size_x, map_size_y):
    ims = floorplan.rgb(states, (map_size_x, map_size_y))
    for i, im in enumerate(ims):
        cv2.circle(im, (im.shape[1]/2, im.shape[0]/2), 1, (0, 0, 255), -1)
        cv2.imwrite('floorplan_%d.png' % (i), im)


def plot_att_graph(act_att, obs_att, seq, seq_len, black_background=False):

    if black_background:
        params = {"ytick.color": "w",
                  "xtick.color": "w",
                  "axes.labelcolor": "w",
                  "axes.edgecolor": "w",
                  "figure.facecolor": "k",
                  "axes.facecolor": "k"}
        plt.rcParams.update(params)

    fig = plt.figure(figsize=(5.52,8.88))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=1, hspace=0) # set the spacing between axes.
    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs1[1])
    im1 = ax1.imshow(act_att)
    im2 = ax2.imshow(obs_att)
    ax1.set_yticklabels([""]+seq)
    ax2.set_yticklabels([""]+seq)
    K = act_att.shape[1]
    ax1.xaxis.set_major_locator(plt.MaxNLocator(K))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(K))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(seq_len+2))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(seq_len+2))

    ax1.set_xlabel('Timesteps')
    ax2.set_xlabel('Timesteps')
    # ax1.set_ylabel('Instruction')
    # ax2.set_ylabel('Instruction')
    ax1.set_title('Motion Attention', y=1.15, color='w' if black_background else 'k')
    ax2.set_title('Observation Attention', y=1.15, color='w' if black_background else 'k')

    tick_locator = ticker.MaxNLocator(nbins=4)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('top', size='1%', pad=0.5)
    cb = fig.colorbar(im1, cax=cax, orientation='horizontal')
    cb.locator = tick_locator
    cb.ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    cb.update_ticks()

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('top', size='1%', pad=0.5)
    cb = fig.colorbar(im2, cax=cax, orientation='horizontal')
    cb.locator = tick_locator
    cb.ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    cb.update_ticks()

    # ax1.set_aspect('equal')
    # ax2.set_aspect('equal')
    if black_background:
        params = {"ytick.color": "k",
                  "xtick.color": "k",
                  "axes.labelcolor": "k",
                  "axes.edgecolor": "k",
                  "figure.facecolor": "w",
                  "axes.facecolor": "w"}
        plt.rcParams.update(params)

    # fig.suptitle("Attention over Instruction Words", fontsize=16)
    # fig.tight_layout()
    return fig


def figure_to_rgb(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()       # draw the canvas, cache the renderer
    size = fig.get_size_inches()*fig.dpi
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(size[1]),int(size[0]),3)
    return image
