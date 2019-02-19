#!/bin/bash

export PATH_VID2DEPTH=$(pwd)
export PATH_KITTI=$(pwd)/data

python $PATH_VID2DEPTH/evaluate.py \
    --split kitti \
    --prediction_path $PATH_VID2DEPTH/inference-validation/ \
    --gt_path $PATH_KITTI/2011_09_26/2011_09_26_drive_0002_sync/oxts/data

