#!/bin/bash

export PATH_VID2DEPTH=$(pwd)
export PATH_KITTI_ODOM=$(pwd)/data

python $PATH_VID2DEPTH/kitti_eval/eval_pose.py \
    --pred_dir $PATH_VID2DEPTH/inference_validation_egomotion/ \
    --gtruth_dir $PATH_KITTI_ODOM/pose_data_processed/ground_truth_processed/ 

#    --gtruth_dir $PATH_KITTI_ODOM/pose_data_processed/ground_truth/
#python $PATH_VID2DEPTH/kitti_eval/evaluate.py \
#    --split kitti \
#    --prediction_path $PATH_VID2DEPTH/inference_validation_egomotion/ \
#    --gt_path $PATH_KITTI_ODOM/sequences/10
