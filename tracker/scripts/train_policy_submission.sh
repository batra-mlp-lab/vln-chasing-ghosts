#!/bin/sh

cd /root/mount/Matterport3DSimulator
python3 tracker/train_policy.py -visdom -preloading -train_split trainval -exp_name Release_Policy_submission -validate_every 500 -validation_iterations 100 -visdom_server http://asimo.cc.gatech.edu -visdom_port 8097
