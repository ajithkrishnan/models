#!/bin/bash

python2 $PATH_VID2DEPTH/dataset/gen_data.py \
      --dataset_name kitti_raw_eigen \
      --dataset_dir /raid/data/Datasets/PublicDatasets/KITTI/kitti-raw-uncompressed \
      --data_dir $PATH_VID2DEPTH/data/kitti_raw_eigen \
      --seq_length 3
