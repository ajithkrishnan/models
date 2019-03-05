import numpy as np
import os
#import cv, cv2
import argparse
import tensorflow as tf
import csv
#from evaluation_utils import *

parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
parser.add_argument('--split',               type=str,   help='data split, kitti or eigen',         required=True)
parser.add_argument('--prediction_path', type=str,   help='path to estimated disparities',      required=True)
parser.add_argument('--gt_path',             type=str,   help='path to ground truth disparities',   required=True)
parser.add_argument('--min_threshold',           type=float, help='minimum threshold for depth/pose evaluation',        default=1e-3)
parser.add_argument('--max_threshold',           type=float, help='maximum threshold for depth/pose evaluation',        default=80)
#parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14',   action='store_true')

args = parser.parse_args()
gfile = tf.gfile

if __name__ == '__main__':

    if args.split == 'kitti':
        NUM_SAMPLES = 447
        
        if os.path.exists(args.prediction_path) and os.path.exists(args.gt_path):
            with open(os.path.join(args.prediction_path, 'inference.txt')) as pred_file:
                                
                    pred_reader = csv.reader(pred_file, delimiter=' ')
                    row_count = 0 
                    gt_prev_pose = []
                    ate_all = []

                    for pred_row, gt_file in zip(pred_reader, sorted(os.listdir(args.gt_path))):
                        with open(pred_row[0]) as pred_pose_file, open(os.path.join(args.gt_path, gt_file)) as gt_pose_file:

                            pred_pose_reader = csv.reader(pred_pose_file, delimiter=' ')
                            gt_pose_reader = csv.reader(gt_pose_file, delimiter=' ')                            
                            #    DEBUG: To print the no. of rows in the file
                            #    a = sum(1 for line in gt_pose_file)

                            for delta_pose, gt_pose in zip(pred_pose_reader, gt_pose_reader):
                                gt_curr_pose = np.array([float(value) for value in gt_pose[0:6]])
                                delta_pose = np.array([float(value) for value in delta_pose])
#                                gt_curr_pose = [float(value) for value in gt_pose[0:6]]
#                                delta_pose = [float(value) for value in delta_pose]

                                if row_count == 0:
                                    gt_prev_pose = gt_curr_pose
                                else:
#                                    pred_pose = gt_prev_pose + delta_pose
                                    pred_pose = np.add(gt_prev_pose, delta_pose)
                                    scale = np.sum(gt_curr_pose * pred_pose)/np.sum(pred_pose ** 2)
                                    alignment_error = pred_pose * scale - gt_curr_pose 
#                                    rmse = np.sqrt(np.sum(alignment_error ** 2)))/len()
                                    rmse = np.sqrt(np.sum(alignment_error ** 2))/ NUM_SAMPLES
                                    ate_all.append(rmse)
                                    gt_prev_pose = gt_curr_pose

#                                    print("ROW Count {}".format(row_count))
#                                    print("gt_curr_pose : {}".format(gt_curr_pose[0]))
#                                    print("pred_pose lat: {}".format(pred_pose[0]))

#
                            row_count += 1

                    ate_all = np.array(ate_all)
                    print("ATE mean: {}, std: {}".format(np.mean(ate_all), np.std(ate_all)))

    


                #pose_data = pred_egomotion_f.readlines()
                #print(pose_data)


#            for filename in gfile.ListDirectory(args.prediction_path):
#                gt_egomotion = gfile.Open(filename,'r')
#                # DEBUG
#
#                filename = os.path.join(args.prediction_path, filename)
#                #print(filename)
#                gt_egomotion = pd.read_csv(filename, sep=' ', header=None)
#                gt_egomotion = pd.read_csv('/home/trn_ak/git_clones/models/research/vid2depth/inference_96560_egomotion/03320001.txt', sep=' ', header=None)
#
#                #print(gt_egomotion[0][0])
#
#    elif args.split == 'eigen':
#        num_samples = 697
#        test_files = read_text_lines(args.gt_path + 'eigen_test_files.txt')
#        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args.gt_path)
#
#        num_test = len(im_files)
#        gt_depths = []
#        pred_depths = []
#        for t_id in range(num_samples):
#            camera_id = cams[t_id]  # 2 is left, 3 is right
#            depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
#            gt_depths.append(depth.astype(np.float32))
#
#            disp_pred = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_LINEAR)
#            disp_pred = disp_pred * disp_pred.shape[1]
#
#            # need to convert from disparity to depth
#            focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
#            depth_pred = (baseline * focal_length) / disp_pred
#            depth_pred[np.isinf(depth_pred)] = 0
#
#            pred_depths.append(depth_pred)
#
#   rms     = np.zeros(num_samples, np.float32)
#   log_rms = np.zeros(num_samples, np.float32)
#   abs_rel = np.zeros(num_samples, np.float32)
#   sq_rel  = np.zeros(num_samples, np.float32)
#   d1_all  = np.zeros(num_samples, np.float32)
#   a1      = np.zeros(num_samples, np.float32)
#   a2      = np.zeros(num_samples, np.float32)
#   a3      = np.zeros(num_samples, np.float32)
#   
#    for i in range(num_samples):
#        
#        gt_depth = gt_depths[i]
#        pred_depth = pred_depths[i]
#
#        pred_depth[pred_depth < args.min_depth] = args.min_depth
#        pred_depth[pred_depth > args.max_depth] = args.max_depth
#
#        if args.split == 'eigen':
#            mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)
#
#            
#            if args.garg_crop or args.eigen_crop:
#                gt_height, gt_width = gt_depth.shape
#
#                # crop used by Garg ECCV16
#                # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
#                if args.garg_crop:
#                    crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
#                                     0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
#                # crop we found by trial and error to reproduce Eigen NIPS14 results
#                elif args.eigen_crop:
#                    crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,   
#                                     0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
#
#                crop_mask = np.zeros(mask.shape)
#                crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
#                mask = np.logical_and(mask, crop_mask)
#
#        if args.split == 'kitti':
#            gt_disp = gt_disparities[i]
#            mask = gt_disp > 0
#            pred_disp = pred_disparities_resized[i]
#
#            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
#            bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
#            d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()
#
#        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])
#
#    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
#    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
