import numpy as np
import os
import glob
from tqdm import tqdm

class points_range():
    def __init__(self,scene):
        
        if scene == 'lidar':
            scene_dir = '/clever/volumes/gfs-35-42/wrb/road_method_code/lidar'
        elif scene == 'camera':
            scene_dir = '/clever/volumes/gfs-35-42/wrb/road_method_code/camera_offset'
        elif scene == 'sfm_dense':
            scene_dir = '/clever/volumes/gfs-35-30/wrb/road_method_code/sfm_dense'
        elif scene == 'sfm_sparse':
            scene_dir = '/clever/volumes/gfs-35-30/wrb/road_method_code/sfm_sparse'
        data_path = np.sort(glob.glob(os.path.join(scene_dir,'*.npy')))

        max_val_x = 0
        max_val_y = 0
        min_val_x = 999999
        min_val_y = 999999
        for frame_id,points in tqdm(enumerate(data_path)):
            
            world_points = np.load(points)
            max_val = np.max(world_points[:,:2],axis = 0)
            if max_val[0] >= max_val_x:
                max_val_x = max_val[0]
            if max_val[1] >= max_val_y:
                max_val_y = max_val[1]
            min_val = np.min(world_points[:,:2],axis = 0)
            if min_val[0] <= min_val_x:
                min_val_x = min_val[0]
            if min_val[1] <= min_val_y:
                min_val_y = min_val[1]


        self.max_val_x = max_val_x
        self.max_val_y = max_val_y
        self.min_val_x = min_val_x
        self.min_val_y = min_val_y
        
    def get_min_max(self):
        return np.array([self.max_val_x,self.max_val_y]),np.array([self.min_val_x,self.min_val_y])