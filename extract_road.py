import imageio
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='argparse learning')
parser.add_argument('--img_dir', type=str, default='/clever/volumes/gfs-35-30/wangruibo/kitti_odo/kitti_odometry/00/image_2/', help='image direction')
parser.add_argument('--sem_dir', type=str, default='/clever/volumes/gfs-35-30/wangruibo/kitti_odo/kitti_odometry/seg_sequences/00/image_2', help='semantics direction')
parser.add_argument('--img_save_file', type=str, default='image', help='image extract save direction')
parser.add_argument('--sem_save_file', type=str, default='semantic', help='semantic extract save direction')
args = parser.parse_args()

all_sem_path = []
for img in os.listdir(args.img_dir):
    all_sem_path.append(args.img_dir + img)
all_sem_path = np.sort(all_sem_path)

all_imge_path = []
for img in os.listdir(args.sem_dir):
    all_imge_path.append(args.sem_dir + img)
all_imge_path = np.sort(all_imge_path)

if not os.path.exists(args.img_save_file):
    os.makedirs(args.img_save_file)

if not os.path.exists(args.sem_save_file):
    os.makedirs(args.sem_save_file)

for img,sem in tqdm(zip(all_imge_path,all_sem_path)):
    semantic = cv2.imread(sem, cv2.IMREAD_UNCHANGED)
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    image[np.where((semantic != 13) & (semantic != 24) & (semantic != 41))] = [0,0,0]
    imageio.imwrite(args.img_save_file + '/' + img.split('/')[-1], image)

for sem in tqdm(all_sem_path):
    semantic = cv2.imread(sem, cv2.IMREAD_UNCHANGED)
    semantic[np.where((semantic != 13) & (semantic != 24) & (semantic != 41))] = 0
    imageio.imwrite(args.sem_save_file + '/' + sem.split('/')[-1], semantic)