# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Generates ground truth data for evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import model
import numpy as np
import scipy.misc
import tensorflow as tf
import util
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM
import csv

gfile = tf.gfile

HOME_DIR = os.path.expanduser('~')
DEFAULT_OUTPUT_DIR = os.path.join(HOME_DIR, 'vid2depth/inference')
DEFAULT_KITTI_DIR = os.path.join(HOME_DIR, 'kitti-raw-uncompressed')
DEFAULT_MODE = 'depth'

flags.DEFINE_string('output_dir', DEFAULT_OUTPUT_DIR,
                    'Directory to store estimated depth maps.')
flags.DEFINE_string('gt_dir', DEFAULT_KITTI_DIR, 'KITTI dataset directory.')
flags.DEFINE_string('kitti_sequence', None, 'KITTI video directory name.')
flags.DEFINE_integer('seq_length', 3, 'Sequence length for each example.')
flags.DEFINE_string('mode', DEFAULT_MODE, 'Specify the network to run inference on i.e depth or pose' )

FLAGS = flags.FLAGS

flags.mark_flag_as_required('gt_dir')
flags.mark_flag_as_required('kitti_sequence')
flags.mark_flag_as_required('seq_length')


def _gen_data():

    gt_path = os.path.join(FLAGS.kitti_dir, '%.2d_full.txt' % FLAGS.kitti_sequence)
    if not os.path.exists(gt_path) :
        break
    else:
        with open(gt_path) as gt_file:
            gt_reader = list(csv.reader(gt_file, delimiter=' '))

            times = []
            for j, _ in enumerate(gt_reader):
                gt_reader[j] = [float(i) for i in gt_reader[j]]
                times.append(g_row[0])

    #        max_offset = (FLAGS.seq_length - 1)//2
            gt_reader = np.array(gt_reader)
            times = np.array(times)
            test_frames = ['%.2d %.6d' % (int(FLAGS.kitti_sequence), n) for n in range(len(gt_reader))]
            max_offset = 1
          
            for tgt_idx in range(0, len(gt_reader)):

                if not is_valid_sample(test_frames, tgt_idx, FLAGS.seq_length):
                  continue
                if tgt_idx % 100 == 0:
                  logging.info('Generating from %s: %d/%d', gt_path, tgt_idx,
                              len(gt_reader))

                # TODO: currently assuming batch_size = 1

                pose_seq = load_sequence(FLAGS.gt_dir, 
                                                test_frames, 
                                                tgt_idx, 
                                                FLAGS.seq_length)

                egomotion_data = pose_seq
                # Insert target poses
                # DEBUG: check if the target pose is at the right index
                #        egomotion_data = np.insert(egomotion_data, 0, np.zeros((1,6)), axis=0) 
                #        egomotion_data = np.insert(egomotion_data, 2, np.zeros((1,6)), axis=0) 
                egomotion_data = np.insert(egomotion_data, max_offset, np.zeros((1,6)), axis=0) 
                curr_times = times[tgt_idx - max_offset:tgt_idx + max_offset + 1]
                egomotion_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx - max_offset)
                #        egomotion_path = os.path.join(FLAGS.output_dir, str(egomotion_file))
                for j, g_row in enumerate(gt_reader):
                with open(os.path.join(FLAGS.output_dir, '%.6d.txt' % j),'w') as f:
                        writer = csv.writer(f, delimiter=' ')
                #                        writer.writerows(pose_seq)
                        writer.writerow(pose_seq)


def load_sequence(dataset_dir, 
                        frames, 
                        tgt_idx, 
                        seq_length)
#    max_offset = int((seq_length - 1)/2)
    max_offset = 1
    for o in range(-max_offset, max_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        pose_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        curr_img = scipy.misc.imread(img_file)
        if o == -max_offset:
            image_seq = curr_img
        else:
            image_seq = np.hstack((image_seq, curr_img))
    return image_seq


def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)

    if tgt_idx >= N:
      return False
    tgt_drive, _ = frames[tgt_idx].split(' ')
    #TODO: calculate max_offset in a clean way 
#    max_offset = (seq_length - 1)//2
    max_offset = 1
    min_src_idx = tgt_idx - max_offset
    max_src_idx = tgt_idx + max_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False


def main(_):
  _gen_data()
