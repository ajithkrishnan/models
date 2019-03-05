#!/bin/bash

export PATH_KITTI_GT=$(pwd)/../data/pose_data_processed/

python3 process_gtruth.py \
    --output_dir $PATH_KITTI_GT/ground_truth_processed/ \
    --gt_dir $PATH_KITTI_GT/ground_truth/ \
    --kitti_sequence 09 \
    --seq_length 3
