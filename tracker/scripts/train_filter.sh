#!/bin/sh

cd /root/mount/Matterport3DSimulator
python3 tracker/train_filter.py -visdom -preloading -exp_name Release_Filter -visdom_server http://asimo.cc.gatech.edu -visdom_port 8097
