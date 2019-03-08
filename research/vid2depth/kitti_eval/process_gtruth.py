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
flags.DEFINE_boolean('plot', False, 'Sets the mode to plot')

FLAGS = flags.FLAGS

flags.mark_flag_as_required('gt_dir')
flags.mark_flag_as_required('output_dir')
flags.mark_flag_as_required('kitti_sequence')

origin = []

def _gen_data():

    gt_path = os.path.join(FLAGS.gt_dir, 'poses/%.2d.txt' % FLAGS.kitti_sequence)
    output_dir = FLAGS.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    if os.path.exists(gt_path) :
        gt_list = []
        times = []

        with open(gt_path) as gt_file:
            gt_list = list(csv.reader(gt_file, delimiter=' '))

        for j, _ in enumerate(gt_list):
            gt_list[j] = [float(i) for i in gt_list[j]]

#        max_offset = (FLAGS.seq_length - 1)//2
        gt_array = np.array(gt_list)
      
        with open(FLAGS.kitti_dir + 'sequences/%.2d/times.txt' % FLAGS.kitti_sequence, 'r') as f:
            times = f.readlines()
        times = np.array([float(s[:-1]) for s in times])

        test_frames = ['%.2d %.6d' % (int(FLAGS.kitti_sequence), n) for n in range(len(gt_list))]
        max_offset = 1

        if FLAGS.plot:
            tgt_idx = 0
            egomotion_data = load_sequence(FLAGS.gt_dir, 
                                            tgt_idx, 
                                            gt_array, 
                                            FLAGS.seq_length,
                                            FLAGS.plot)
            curr_times = times[0:len(gt_array) + 1]
            num_curr_times = len(times[0:len(gt_array) + 1])

            egomotion_file = output_dir + 'groundtruth.txt' 
            dump_pose_seq_TUM(egomotion_file, egomotion_data, curr_times, FLAGS.plot)

        else:

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

                curr_times = times[tgt_idx:tgt_idx + max_offset + 2]
                egomotion_file = output_dir + '%.6d.txt' % (tgt_idx)
                dump_pose_seq_TUM(egomotion_file, egomotion_data, curr_times)


def main(_):
  _gen_data()

if __name__ == '__main__':
    app.run(main)
