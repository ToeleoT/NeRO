import numpy as np
import glob
import os
from tqdm import tqdm
import csv
import cv2
from tools import *
import argparse


import argparse

parser = argparse.ArgumentParser(description='argparse learning')
parser.add_argument('--data_dir', type=str, default='/clever/volumes/gfs-35-30/wangruibo/kitti_odo/kitti_odometry', help='data dirctor')
parser.add_argument('--save_dir', type=str, default='lidar', help='save dirctory')
parser.add_argument('--pose_dir', type=str, default='00.txt', help='pose dirctory')

args = parser.parse_args()

scene_dir = args.data_dir
P0, P1, P2, P3,  lidar2camera_m,extrinsic_m = read_calib(scene_dir +'/00_calib/calib.txt')

intrinsics = P2[:,:3]

pose_path = os.path.join(scene_dir,args.pose_dir)
poses = read_pose(pose_path)

lidar_path = np.sort(glob.glob(os.path.join(scene_dir,'00_velodyne','velodyne','*.bin')))

folder_path = args.save_dir
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
image_path = np.sort(glob.glob(os.path.join(scene_dir,'train','*.png')))

for file_no,(lidar_bin,pose,image_path) in tqdm(enumerate(zip(lidar_path,poses,image_path))):
    if file_no >= len(image_path):
        break
    lidar = read_bin(lidar_bin)
    image = cv2.imread(image_path)
    point_in_world = generate_road(lidar,image,pose,lidar2camera_m,camera2chassis,intrinsics)
    np.save(os.path.join(folder_path,'%06d.npy' % file_no),point_in_world)