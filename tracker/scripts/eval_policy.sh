#!/bin/bash

cd /root/mount/Matterport3DSimulator
for IT in {2000..15000..500}; do
  echo $IT
  python3 tracker/train_policy.py -visdom -visdom_server http://asimo.cc.gatech.edu -preloading -exp_name Policy -eval -val_epoch $IT
done
