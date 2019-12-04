#!/bin/sh

cd /root/mount/Matterport3DSimulator
python3 tracker/train_policy.py -visdom -preloading -exp_name Release_Policy -visdom_server http://asimo.cc.gatech.edu -visdom_port 8097
