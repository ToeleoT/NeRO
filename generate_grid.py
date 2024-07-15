import numpy as np
import glob
import os
from tqdm import tqdm
import cv2
from tools import *
import argparse


import argparse

parser = argparse.ArgumentParser(description='argparse learning')
parser.add_argument('--data_dir', type=str, default='/clever/volumes/gfs-35-42/wrb/road_method_code', help='data dirctory')
parser.add_argument('--save_dir', type=str, default='camera_offset', help='save dirctory')
parser.add_argument('--pose_dir', type=str, default='00.txt', help='pose dirctory')
parser.add_argument('--range_list', type=list, default=[5,20,-10,10], help='pose dirctory')
parser.add_argument('--size', type=float, default=0.1, help='gird point seperate distance')
parser.add_argument('--height', type=float, default=1.65, help='camera height')
args = parser.parse_args()


scene_dir = args.data_dir
offset_z = np.arange(args.range_list[0],args.range_list[1],args.size)
offset_x = np.arange(args.range_list[2],args.range_list[3],args.size)
offset_y = args.height #kitti camera height
X, Y, Z = np.meshgrid(offset_x,offset_y,offset_z)
camera_points = np.array([i for i in zip(X.flat,Y.flat,Z.flat)])
pose_path = os.path.join(scene_dir,args.pose_dir)
folder_path = args.save_dir
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
c2w = read_pose(pose_path)
image_path = np.sort(glob.glob(os.path.join(scene_dir,'train','*.png')))
for frame_id,c2w_each in tqdm(enumerate(c2w)):
    if frame_id >= len(image_path):
        break
    pose_world = camera2chassis @ c2w_each
    R = pose_world[:3,:3]
    T = pose_world[:3,3]
    world_points = camera_points @ R.T + T
    np.save(os.path.join(folder_path, '%06d.npy'%frame_id),world_points)