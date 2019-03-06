#!/bin/bash

python plot_trajectories/evaluate_ate.py \
    --first_file data/dataset/ground_truth_processed/001549.txt \
    --second_file inference_evaluation_egomotion/001549.txt \
    --plot PLOT \
   --verbose 


