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

args = parser.parse_args()

scene_dir = args.data_dir
P0, P1, P2, P3,  lidar2camera_m,extrinsic_m = read_calib(scene_dir +'/00_calib/calib.txt')

intrinsics = P2[:,:3]

np.savetxt('intrinsics.txt',P2[:,:3])