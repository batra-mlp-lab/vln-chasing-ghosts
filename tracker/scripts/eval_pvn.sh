#!/bin/bash

cd /root/mount/Matterport3DSimulator
for IT in {2000..15000..500}; do
  echo $IT
  python3 tracker/train_pvn.py -visdom -visdom_server http://asimo.cc.gatech.edu -preloading -map_range_x 128 -map_range_y 128 -exp_name PVN -eval -val_epoch $IT
done
