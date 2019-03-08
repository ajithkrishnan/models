#!/bin/bash

export PATH_VID2DEPTH=$(pwd)
export PATH_KITTI_ODOM=$PATH_VID2DEPTH/data/dataset

python3 $PATH_VID2DEPTH/convert.py \
    --input_dir $PATH_KITTI_ODOM/ground_truth_processed/ \
    --output_dir $PATH_KITTI_ODOM/ground_truth_processed/ \
    --file_type groundtruth


