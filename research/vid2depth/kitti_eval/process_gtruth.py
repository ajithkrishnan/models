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

"""Preprocesses ground truth data for evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
#import util
import csv
from process_gtruth_utils import dump_pose_seq_TUM, is_valid_sample, load_sequence
import math

FLOAT_EPS = np.finfo(np.float).eps

HOME_DIR = os.path.expanduser('~')

flags.DEFINE_string('output_dir', None,
                    'Directory to store estimated depth maps.')
flags.DEFINE_string('gt_dir', None, 'KITTI groundtruth directory.')
flags.DEFINE_string('kitti_dir', None, 'KITTI dataset directory.')
flags.DEFINE_integer('kitti_sequence', None, 'KITTI video directory name.')
flags.DEFINE_integer('seq_length', 3, 'Sequence length for each example.')

FLAGS = flags.FLAGS

flags.mark_flag_as_required('gt_dir')
flags.mark_flag_as_required('output_dir')
flags.mark_flag_as_required('kitti_sequence')


def _gen_data():

    # DEBUG
#    gt_path = os.path.join(FLAGS.gt_dir, '%.2d_full.txt' % FLAGS.kitti_sequence)
    gt_path = os.path.join(FLAGS.gt_dir, 'poses/%.2d.txt' % FLAGS.kitti_sequence)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    if os.path.exists(gt_path) :
        gt_list = []
        times = []

        with open(gt_path) as gt_file:
            gt_list = list(csv.reader(gt_file, delimiter=' '))

        for j, _ in enumerate(gt_list):
            gt_list[j] = [float(i) for i in gt_list[j]]
            # DEBUG
#            times.append(gt_list[j][0])

#        max_offset = (FLAGS.seq_length - 1)//2
        gt_array = np.array(gt_list)
        # DEBUG
#        gt_array = np.delete(gt_array, 0 , axis=1)
#        times = np.array(times)
      
        with open(FLAGS.kitti_dir + 'sequences/%.2d/times.txt' % FLAGS.kitti_sequence, 'r') as f:
            times = f.readlines()
        times = np.array([float(s[:-1]) for s in times])

        test_frames = ['%.2d %.6d' % (int(FLAGS.kitti_sequence), n) for n in range(len(gt_list))]
        max_offset = 1

        for tgt_idx in range(0, len(gt_list)):

            if not is_valid_sample(test_frames, tgt_idx, FLAGS.seq_length):
              continue
            if tgt_idx % 100 == 0:
              logging.info('Generating: %d/%d', tgt_idx,
                          len(gt_list))

            # TODO: currently assuming batch_size = 1

            egomotion_data = load_sequence(FLAGS.gt_dir, 
                                            tgt_idx, 
                                            gt_array, 
                                            FLAGS.seq_length)
            # DEBUG
            if tgt_idx % 100 == 0:
                print("shape of egomotion_data: {}".format(egomotion_data.shape))
                print("shape of gt_array: {}".format(gt_array.shape))
            # DEBUG
#            egomotion_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx - max_offset)
#            curr_times = times[tgt_idx - max_offset:tgt_idx + max_offset + 1]
            egomotion_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx)
            curr_times = times[tgt_idx:tgt_idx + max_offset + 2]
            dump_pose_seq_TUM(egomotion_file, egomotion_data, curr_times)


def main(_):
  _gen_data()

if __name__ == '__main__':
    app.run(main)
