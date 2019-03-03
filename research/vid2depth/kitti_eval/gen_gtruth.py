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
import numpy as np
#import util
import csv
from pose_evaluation_utils import pose_vec_to_mat, rot2quat

HOME_DIR = os.path.expanduser('~')

flags.DEFINE_string('output_dir', None,
                    'Directory to store estimated depth maps.')
flags.DEFINE_string('gt_dir', None, 'KITTI dataset directory.')
flags.DEFINE_string('kitti_sequence', None, 'KITTI video directory name.')
flags.DEFINE_integer('seq_length', 3, 'Sequence length for each example.')

FLAGS = flags.FLAGS

flags.mark_flag_as_required('gt_dir')
flags.mark_flag_as_required('output_dir')
flags.mark_flag_as_required('kitti_sequence')


def _gen_data():

    gt_path = os.path.join(FLAGS.gt_dir, '%.2d_full.txt' % int(FLAGS.kitti_sequence))
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    if os.path.exists(gt_path) :
        gt_list = []
        times = []

        with open(gt_path) as gt_file:
            gt_list = list(csv.reader(gt_file, delimiter=' '))

        for j, _ in enumerate(gt_list):
            gt_list[j] = [float(i) for i in gt_list[j]]
            times.append(gt_list[j][0])

#        max_offset = (FLAGS.seq_length - 1)//2
        gt_array = np.array(gt_list)
        gt_array = np.delete(gt_array, 0 , axis=1)
        times = np.array(times)
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

            # Insert target poses
            # DEBUG: check if the target pose is at the right index
            #        egomotion_data = np.insert(egomotion_data, 0, np.zeros((1,6)), axis=0) 
            #        egomotion_data = np.insert(egomotion_data, 2, np.zeros((1,6)), axis=0) 
#            zero_pose = np.zeros((1,8))
#            zero_pose[0][0] = gt_array[tgt_idx][0]
            zero_pose = np.zeros((1,7))
            egomotion_data = np.insert(egomotion_data, max_offset, zero_pose, axis=0) 
            # DEBUG
            if tgt_idx % 100 == 0:
                print("shape of egomotion_data: {}".format(egomotion_data.shape))
                print("shape of gt_array: {}".format(gt_array.shape))
            curr_times = times[tgt_idx - max_offset:tgt_idx + max_offset + 1]
            egomotion_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx - max_offset)
            dump_pose_seq_TUM(egomotion_file, egomotion_data, curr_times)
#            #        egomotion_path = os.path.join(FLAGS.output_dir, str(egomotion_file))
#            for j, g_row in enumerate(gt_list):
#                with open(os.path.join(FLAGS.output_dir, '%.6d.txt' % j),'w') as f:
#                    writer = csv.writer(f, delimiter=' ')
#    #                writer.writerows(pose_seq)
#                    writer.writerow(pose_seq)


def dump_pose_seq_TUM(out_file, poses, times):
    # Set first frame as the origin
    first_origin = pose_vec_to_mat(poses[0])
    with open(out_file, 'w') as f:
        for p in range(len(times)):
            this_pose = pose_vec_to_mat(poses[p])
            this_pose = np.dot(first_origin, np.linalg.inv(this_pose))
            tx = this_pose[0, 3]
            ty = this_pose[1, 3]
            tz = this_pose[2, 3]
            rot = this_pose[:3, :3]
            qw, qx, qy, qz = rot2quat(rot)
            f.write('%f %f %f %f %f %f %f %f\n' % (times[p], tx, ty, tz, qx, qy, qz, qw))

def load_sequence(dataset_dir, 
                        tgt_idx, 
                        gt_array, 
                        seq_length):
#    max_offset = int((seq_length - 1)/2)
    max_offset = 1
#    for o in range(-max_offset, max_offset+1):
    # DEBUG: Dirty Fix
    for o in range(0, 2):
        curr_idx = tgt_idx + o
        curr_pose = gt_array[curr_idx]
#        if o == -max_offset:
        if o == 0:
            pose_seq = curr_pose 
        else:
            pose_seq = np.vstack((pose_seq, curr_pose))
    return pose_seq


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

if __name__ == '__main__':
    app.run(main)
