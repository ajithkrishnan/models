#!/bin/bash

python plot_trajectories/evaluate_ate.py \
    --first_file data/pose_data_processed/ground_truth/09_full.txt \
    --second_file ~/tensorflow/thesis/SfMLearner/test_output_1103/plot/inference.txt \
    --plot SFM_PLOT_1103 \
    --verbose 


