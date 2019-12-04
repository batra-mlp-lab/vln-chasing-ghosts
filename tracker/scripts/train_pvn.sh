#!/bin/sh

cd /root/mount/Matterport3DSimulator
python3 tracker/train_pvn.py -visdom -preloading -map_range_y 128 -map_range_x 128 -belief_downsample_factor 1 -batch_size 4 -exp_name Release_PVN -visdom_server http://asimo.cc.gatech.edu -visdom_port 8097
