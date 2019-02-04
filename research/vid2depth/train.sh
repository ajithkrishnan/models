#!/bin/bash

python2 $PATH_VID2DEPTH/train.py \
      --data_dir $PATH_VID2DEPTH/data/kitti_raw_eigen \
      --seq_length 3 \
      --reconstr_weight 0.85 \
      --smooth_weight 0.05 \
      --ssim_weight 0.15 \
      --icp_weight 0 \
      --checkpoint_dir $PATH_VID2DEPTH/checkpoints 
#      --pretrained_ckpt $PATH_VID2DEPTH/checkpoints/model-29465 \



