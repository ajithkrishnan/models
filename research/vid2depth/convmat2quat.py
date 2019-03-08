#!/usr/bin/python

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from absl import app
from absl import flags
import os
import glob
import csv
import numpy as np

"""Converts set of egomotion file sequences into single list"""

flags.DEFINE_string("input_dir", None, "Directory containing sequence files.")
flags.DEFINE_string("output_dir", None, "Directory to which the processed output must be saved.")
flags.DEFINE_string("file_type", None, "Specify if the data is groundtruth or prediction")

FLAGS = flags.FLAGS

def _run_conversion():
    
    pose_full_seq = np.zeros((1,6))
    fixed_origin = np.zeros((1,6))
    times = []

    pose_files = sorted(glob.glob(FLAGS.output_dir + "*.txt"))

    # Number of files
    num_files = len(pose_files)

    if FLAGS.file_type == "groundtruth":
        for seq in range(0, num_files - 1):

            with open(FLAGS.output_dir + "groundtruth.txt", 'a') as out_file :
                writer = csv.writer(out_file, delimiter=' ')
                with open(pose_files[seq],'r') as pose_file:
                    pose_seq = np.array(list(csv.reader(pose_file, delimiter=' ')))
    #               pose_seq = np.delete(pose_seq, 0, axis=1)

                if seq == 0:
    #               fixed_origin = pose_seq[0, :]
    #               pose_full_seq = pose_seq[:, 1:]
    #               times.append(pose_seq[:, 0])
    #                pose_full_seq = pose_seq
                    for i in range(0, pose_seq.shape[0] - 1):
                        writer.writerow(pose_seq[i])

                else:
    #                times.append(pose_seq[FLAGS.seq_length-1, 0])
    #                pose_full_seq = np.vstack((pose_full_seq, pose_seq[FLAGS.seq_length-1, :]))
    #                print(pose_seq[FLAGS.seq_length-1].shape)
    #                pose_full_seq = np.vstack((pose_full_seq, pose_seq[FLAGS.seq_length -1]))
                    writer.writerow(pose_seq[FLAGS.seq_length -1])


#    print(str(pose_full_seq.shape))
#    print(len(times))

    with open(FLAGS.input_dir + "poses/09.txt",  'r') as seq_file:
        with open(FLAGS.input_dir + "sequences/09/times.txt",  'r') as times_file:
            seq_array = np.array(list(csv.reader(seq_file, delimiter=' ')))
            times_array = np.array(list(csv.reader(times_file, delimiter=' ')))


def main(_):
    _run_conversion()

if __name__ == '__main__':
    app.run(main)


