#!/bin/bash

python2 $PATH_VID2DEPTH/inference.py \
      --kitti_video $PATH_KITTI/2011_09_26/2011_09_26_drive_0009_sync \
      --kitti_dir $PATH_KITTI \
      --output_dir $PATH_VID2DEPTH/inference-validation \
      --model_ckpt $PATH_VID2DEPTH/trained_model/model-119496

#      --model_ckpt $PATH_VID2DEPTH/checkpoints/model-32136
#      --input_list_file $PATH_VID2DEPTH/data/kitti_raw_eigen/val.txt \
