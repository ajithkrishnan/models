#!/bin/bash

python plot_trajectories/evaluate_ate.py \
    --first_file data/dataset/ground_truth_processed_plot/groundtruth.txt \
    --second_file inference_evaluation_egomotion/inference.txt \
    --plot PLOT_full \
   --verbose 


