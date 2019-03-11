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

"""Generates egomotion estimates for an entire KITTI visual odom dataset."""

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
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM, is_valid_sample, load_image_sequence

gfile = tf.gfile

HOME_DIR = os.path.expanduser('~')
DEFAULT_OUTPUT_DIR = os.path.join(HOME_DIR, 'vid2depth/inference')
DEFAULT_KITTI_DIR = os.path.join(HOME_DIR, 'kitti-raw-uncompressed')
DEFAULT_MODE = 'depth'

flags.DEFINE_string('output_dir', DEFAULT_OUTPUT_DIR,
                        'Directory to store estimated depth maps.')
flags.DEFINE_string('kitti_dir', DEFAULT_KITTI_DIR, 'KITTI dataset directory.')
flags.DEFINE_string('model_ckpt', None, 'Model checkpoint to load.')
flags.DEFINE_integer('kitti_sequence', None, 'KITTI video directory name.')
flags.DEFINE_integer('batch_size', 4, 'The size of a sample batch.')
flags.DEFINE_integer('img_height', 128, 'Image height.')
flags.DEFINE_integer('img_width', 416, 'Image width.')
flags.DEFINE_integer('seq_length', 3, 'Sequence length for each example.')
flags.DEFINE_string('mode', DEFAULT_MODE, 'Specify the network to run inference on i.e depth or pose' )
flags.DEFINE_boolean('plot', False, 'Set format of output file')

FLAGS = flags.FLAGS

flags.mark_flag_as_required('kitti_sequence')
flags.mark_flag_as_required('model_ckpt')


def _run_egomotion_test():
  """Runs all images through depth model and saves depth maps."""
  ckpt_basename = os.path.basename(FLAGS.model_ckpt)
  ckpt_modelname = os.path.basename(os.path.dirname(FLAGS.model_ckpt))

  fixed_origin = np.zeros((1,6))
  
  if FLAGS.plot:
      output_dir = FLAGS.output_dir + "plot/"
  else:
      output_dir = FLAGS.output_dir  

  if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)
  inference_model = model.Model(is_training=False,
                                seq_length=FLAGS.seq_length,
                                batch_size=FLAGS.batch_size,
                                img_height=FLAGS.img_height,
                                img_width=FLAGS.img_width)
  vars_to_restore = util.get_vars_to_restore(FLAGS.model_ckpt)
  saver = tf.train.Saver(vars_to_restore)
  sv = tf.train.Supervisor(logdir='/tmp/', saver=None)


  with sv.managed_session() as sess:
    saver.restore(sess, FLAGS.model_ckpt)
    if FLAGS.kitti_sequence == 'test_files_eigen':
      im_files = util.read_text_lines(
          util.get_resource_path('dataset/kitti/test_files_eigen.txt'))
      im_files = [os.path.join(FLAGS.kitti_dir, f) for f in im_files]
    else:
      video_path = os.path.join(FLAGS.kitti_dir, 'sequences/%.2d/' % FLAGS.kitti_sequence)
      im_files = gfile.Glob(os.path.join(video_path, 'image_2/', '*.png'))
      im_files = [f for f in im_files if 'disp' not in f]
      im_files = sorted(im_files)

    egomotion_file = None    

    max_offset = (FLAGS.seq_length - 1)//2
#    max_offset = 1
    test_frames = ['%.2d %.6d' % (FLAGS.kitti_sequence, n) for n in range(len(im_files))]
    with open(FLAGS.kitti_dir + 'sequences/%.2d/times.txt' % FLAGS.kitti_sequence, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])
  
    for tgt_idx in range(0, len(im_files)):

        if not is_valid_sample(test_frames, tgt_idx, FLAGS.seq_length):
          continue
        if tgt_idx % 100 == 0:
          logging.info('Generating from %s: %d/%d', ckpt_basename, tgt_idx,
                      len(im_files))

        inputs = np.zeros(
            (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width *  FLAGS.seq_length, 3),
            dtype=np.uint8)
        

    # TODO: currently assuming batch_size = 1

        image_seq = load_image_sequence(FLAGS.kitti_dir, 
                                        test_frames, 
                                        tgt_idx, 
                                        FLAGS.seq_length, 
                                        FLAGS.img_height, 
                                        FLAGS.img_width)

        results = inference_model.inference(image_seq[None,:,:,:], sess, mode=FLAGS.mode)

        egomotion_data = results['egomotion'][0]
        curr_times = times[tgt_idx - max_offset:tgt_idx + max_offset + 1]
        egomotion_data = np.insert(egomotion_data, max_offset, np.zeros((1,6)), axis=0) 
        if FLAGS.plot: 
            if (tgt_idx - 1) == 0:
                fixed_origin = egomotion_data[0]
            egomotion_file = output_dir + 'inference.txt' 
            dump_pose_seq_TUM(egomotion_file, egomotion_data, curr_times, fixed_origin, tgt_idx, FLAGS.plot)
        else:
            egomotion_file = output_dir + '%.6d.txt' % (tgt_idx - max_offset)
            dump_pose_seq_TUM(egomotion_file, egomotion_data, curr_times, np.zeros((1,6)), tgt_idx)
        #DEBUG
#        if tgt_idx % 100 == 0:
#            print("shape of image_seq: {}".format(image_seq.shape))
#            print("shape of results['egomotion']: {}".format(results['egomotion'].shape))
#            print("shape of results['egomotion'][0]: {}".format(results['egomotion'][0].shape))
#            print("shape of egomotion_data: {}".format(egomotion_data.shape))
#            print("shape of curr_times: {}".format(curr_times.shape))


def main(_):
  _run_egomotion_test()


if __name__ == '__main__':
  app.run(main)
