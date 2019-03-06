#!/bin/bash

export PATH_VID2DEPTH=$(pwd)
export PATH_KITTI=$(pwd)/data
export PATH_KITTI_ODOM=$(pwd)/data/dataset

python3 $PATH_VID2DEPTH/test_kitti_pose.py \
      --kitti_dir $PATH_KITTI_ODOM/ \
      --output_dir $PATH_VID2DEPTH/inference_evaluation_egomotion/ \
      --kitti_sequence 09  \
      --model_ckpt $PATH_VID2DEPTH/trained_model/model-119496 \
      --mode egomotion \
      --batch_size 1

#      --model_ckpt $PATH_VID2DEPTH/checkpoints/model-32136
#      --input_list_file $PATH_VID2DEPTH/data/kitti_raw_eigen/val.txt \
#      --kitti_dir $PATH_KITTI \
#      --kitti_video 2011_09_26/2011_09_26_drive_0009_sync \
