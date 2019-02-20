#!/bin/bash

python $PATH_VID2DEPTH/train.py \
      --data_dir $PATH_VID2DEPTH/data/kitti_raw_eigen \
      --seq_length 3 \
      --reconstr_weight 0.85 \
      --smooth_weight 0.05 \
      --ssim_weight 0.15 \
      --icp_weight 0.0 \
      --train_steps 119496 \
      --batch_size 4 \
      --checkpoint_dir $PATH_VID2DEPTH/checkpoints_icp

#      --pretrained_ckpt $PATH_VID2DEPTH/checkpoints_default/last_model \


