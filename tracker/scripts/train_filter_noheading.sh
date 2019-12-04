#!/bin/sh

cd /root/mount/Matterport3DSimulator
python3 tracker/train_filter.py -visdom -preloading -supervision_prob 0.5 -heading_states 1 -exp_name Release_Filter_NoHeading -visdom_server http://asimo.cc.gatech.edu -visdom_port 8097
