from __future__ import division
import os
import numpy as np
import argparse
from glob import glob
from pose_evaluation_utils import *
from absl import logging
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--gtruth_dir", type=str, 
    help='Path to the directory with ground-truth trajectories')
parser.add_argument("--pred_dir", type=str, 
    help="Path to the directory with predicted trajectories")
args = parser.parse_args()

def main():
    pred_files = glob(args.pred_dir + '/*.txt')
    ate_all = []
#    if not os.path.exists(args.gtruth_processed_dir):
#        os.makedirs(args.gtruth_processed_dir)
#    with open(os.path.join(args.gtruth_dir, '09_full.txt')) as gtruth_file,open(os.path.join(args.pred_dir, 'inference.txt')) as inf_file:
#        gtruth_reader = list(csv.reader(gtruth_file, delimiter=' '))
#        inf_reader = csv.reader(inf_file, delimiter=' ')
#        gtruth_value = None
#    
#        for j, g_row in enumerate(gtruth_reader):
#            g_row = [float(i) for i in g_row]
#
#            with open(os.path.join(args.gtruth_processed_dir, '%.6d.txt' % j),'w') as f:
#                    writer = csv.writer(f, delimiter=' ')
#                    writer.writerow(g_row)
    for i in range(len(pred_files)):
        gtruth_file = args.gtruth_processed_dir + os.path.basename(pred_files[i])
        if not os.path.exists(gtruth_file):
            continue
        ate = compute_ate(gtruth_file, pred_files[i])
        if ate == False:
            continue
        ate_all.append(ate)
    ate_all = np.array(ate_all)
    print("Predictions dir: %s" % args.pred_dir)
    print("ATE mean: %.4f, std: %.4f" % (np.mean(ate_all), np.std(ate_all)))
main()
