import numpy as np
import glob
import os
from tqdm import tqdm
import csv
import cv2
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def read_calib(calib_path):
    
    with open(calib_path, 'r') as f:
        raw = f.readlines()
    P0 = np.array(list(map(float, raw[0].split()[1:]))).reshape((3, 4))
    P1 = np.array(list(map(float, raw[1].split()[1:]))).reshape((3, 4))
    P2 = np.array(list(map(float, raw[2].split()[1:]))).reshape((3, 4))
    P3 = np.array(list(map(float, raw[3].split()[1:]))).reshape((3, 4))
    R0 = np.array(list(map(float, raw[4].split()[1:]))).reshape((3, 4))

    R0 = np.vstack((R0, np.array([0, 0, 0, 1])))
    lidar2camera_m = np.array(list(map(float, raw[4].split()[1:]))).reshape((3, 4))
    lidar2camera_m = np.vstack((lidar2camera_m, np.array([0, 0, 0, 1])))

    extrinsic_m = np.matmul(R0, lidar2camera_m)
    return P0, P1, P2, P3,  lidar2camera_m,extrinsic_m

def read_bin(bin_path, intensity=False):
    lidar_points = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    if not intensity:
        lidar_points = lidar_points[:, :3]
    return lidar_points

def read_pose(path):
    if('txt' in path):
        pass
    else:
        path = path + '.txt'
    with open(path, 'r') as csv_file:
        reader = list(csv.reader(csv_file, delimiter=' '))

    poses = []
    for j in range(len(reader) ):
        pose = [float(i) for i in reader[j] ]
        pose = np.array(pose, dtype=np.float32).reshape(3, 4)
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        poses.append(pose)
    return poses


def asvoid(arr):
    """
    Based on http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
    View the array as dtype np.void (bytes). The items along the last axis are
    viewed as one value. This allows comparisons to be performed which treat
    entire rows as one value.
    """
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        """ Care needs to be taken here since
        np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
        Adding 0. converts -0. to 0.
        """
        arr += 0.
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


camera2chassis = np.asarray([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)


def generate_road(point_in_lidar,image,pose,lidar2camera_m,camera2chassis,intrinsics):

    pose_world = camera2chassis @ pose @ lidar2camera_m
    R = pose_world[:3,:3]
    T = pose_world[:3,3]
    point_in_world = point_in_lidar @ R.T + T

    world_cam = np.linalg.inv(pose) @ np.linalg.inv(camera2chassis)
    camera_point = point_in_world @ world_cam[:3,:3].T + world_cam[:3,3]
    camera_point_index = np.argwhere(camera_point[..., 2] > 0.0).flatten()
    camera_point = camera_point[camera_point_index]
    point_in_world = point_in_world[camera_point_index]
    intrinsics = intrinsics[:3,:3]
    u = camera_point[..., 0] / camera_point[..., 2] * intrinsics[0, 0] + intrinsics[0, 2]
    v = camera_point[..., 1] / camera_point[..., 2] * intrinsics[1, 1] + intrinsics[1, 2]
    u_v = np.concatenate((np.expand_dims(u, axis=1),np.expand_dims(v, axis=1)),axis = 1)
    u_v = u_v.astype('int')
    index_need = np.argwhere(((u_v[:,0] >= 0) & (u_v[:,0] < image.shape[1])) & ((u_v[:,1] >= 0) & (u_v[:,1] < image.shape[0]))).flatten()
    u_v = u_v[index_need]
    point_in_world = point_in_world[index_need]
    no_black = np.column_stack(np.where(np.any(image != [0, 0, 0], axis=-1)))
    no_black_index = np.flatnonzero(np.in1d(asvoid(u_v[:,::-1].astype('int')), asvoid(no_black.astype('int'))))
    point_in_world = point_in_world[no_black_index]
    
    return point_in_world


def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)

def calculate_segmentation_metrics(true_labels, predicted_labels, number_classes):

    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    
    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(number_classes)))
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask

    class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
    ious = np.zeros(number_classes)
    for class_id in range(number_classes):
        ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
    miou = nanmean(ious)
    miou_valid_class = np.mean(ious[exsiting_class_mask])
    return miou, miou_valid_class, total_accuracy, class_average_accuracy, ious



def draw_lidar(points, canvas=None, max_dist=None):
    tr = np.asarray([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    points[:,:3] = (tr[:3,:3] @ points[:,:3].T).T
    color_points = points
#     points = points[:,:3]
    points = points[:,:3] - [points[:,0].min(), points[:,1].min(), 0]
    color_points[:,:3] = points
    max_dist = max(
        color_points[:,0].max() - color_points[:,0].min(),
        color_points[:,1].max() - color_points[:,1].min()
    )
    size = 1024
    if canvas is None:
#         canvas = np.zeros([size,size,3]).astype('uint8')
        canvas = np.repeat([255,255,255],size * size).reshape(size,size,3).astype('uint8')
        points_color = [0,0,0]
    else:
        canvas = cv2.resize(canvas,(size,size))
        points_color = [0,0,0]
    ratio = canvas.shape[0]/max_dist
    # offset = np.array([canvas.shape[0], canvas.shape[1]])/2
    offset = 0
    draw_points = (color_points[:,:2]*ratio+offset).astype(int)
    color_points[:,:2] = draw_points
    color_points = color_points[
        (color_points[:,0]>0) & (color_points[:,0]<canvas.shape[0]) &
        (color_points[:,1]>0) & (color_points[:,1]<canvas.shape[1])
    ]
    for sdraw in color_points:
        color = sdraw[3:].astype('uint8')
        canvas[sdraw[0].astype('int'), sdraw[1].astype('int'), :] = color
    return canvas 