#!/usr/bin/python

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from absl import app
from absl import flags
import os
import csv
import numpy as np

"""Converts set of egomotion file sequences into single list"""

flags.DEFINE_string("input_dir", None, "Directory containing sequence files.")
flags.DEFINE_string("output_dir", None, "Directory to which the processed output 
                                         must be saved.")
flags.DEFINE_integer("seq_length", 3, "Number of frames per sequence.")

FLAGS = flags.FLAGS

_run_conversion():
    
    pose_full_seq = np.zeros((1,6))
    fixed_origin = np.zeros((1,6))
    times = []

    pose_files = sorted(glob(FLAGS.output_dir + "*.txt"))

    # Number of files
    num_files = len(pose_files)

    for i in range(0, num_files + 1):
        with open(pose_files[i],'r') as pose_file:
           pose_seq = np.array(list(csv.reader(pose_file, delimiter=' ')))
           times.append(pose_seq[:][0])
           pose_seq = np.delete(pose_seq, 0, axis=1)
           pose_full_seq = np.vstack((pose_full_seq, pose_seq))

           if i == 0:
               fixed_origin = pose_seq[0][:]




def main(_):
    _run_conversion()

if __name__ == '__main__':
    app.run(main)


