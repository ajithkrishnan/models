#!/bin/bash

#export PATH_VID2DEPTH=$(pwd)
#export PATH_KITTI=$(pwd)/data

python $PATH_VID2DEPTH/inference.py \
      --kitti_dir $PATH_KITTI_ODOM/ \
      --output_dir $PATH_VID2DEPTH/inference_validation_egomotion/ \
      --kitti_video 09  \
      --model_ckpt $PATH_VID2DEPTH/trained_model/model-119496 \
      --mode egomotion 

#      --model_ckpt $PATH_VID2DEPTH/checkpoints/model-32136
#      --input_list_file $PATH_VID2DEPTH/data/kitti_raw_eigen/val.txt \
#      --kitti_dir $PATH_KITTI \
#      --kitti_video 2011_09_26/2011_09_26_drive_0009_sync \
