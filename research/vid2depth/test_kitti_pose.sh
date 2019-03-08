#!/bin/bash

export PATH_VID2DEPTH=$(pwd)
export PATH_KITTI_ODOM=$(pwd)/data/dataset

python3 $PATH_VID2DEPTH/test_kitti_pose.py \
      --kitti_dir $PATH_KITTI_ODOM/ \
      --output_dir $PATH_VID2DEPTH/inference_evaluation_egomotion/ \
      --plot True \
      --kitti_sequence 09  \
      --model_ckpt $PATH_VID2DEPTH/trained_model/model-119496 \
      --mode egomotion \
      --batch_size 1 



