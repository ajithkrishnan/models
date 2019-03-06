#!/bin/bash

#export PATH_KITTI_GT=$(pwd)/../data/pose_data_processed/
export PATH_KITTI_GT=$(pwd)/data/dataset

python3 kitti_eval/process_gtruth.py \
    --output_dir $PATH_KITTI_GT/ground_truth_processed/ \
    --kitti_dir $PATH_KITTI_GT/ \
    --gt_dir $PATH_KITTI_GT/ \
    --kitti_sequence 09 \
    --seq_length 3 
