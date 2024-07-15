import os
import glob
import numpy as np
from skimage.io import imread
import cv2
import imageio
import json
import os
import pandas as pd
from tqdm import tqdm
from tools import *


kitti_color_code = np.array([(128,64,128),(244,35,232),(153,153,153),(111,74,0)]).astype(np.uint8)


class kitti_Dataset(object):
    def __init__(self, scene_dir,pose_path):
        self.scene_dir = scene_dir
        self.semantic_class_dir = scene_dir

        intrinsic = np.loadtxt(os.path.join(scene_dir,"intrinsics.txt"))
        intrinsics = [
            [intrinsic[0,0], 0, intrinsic[0,2]],
            [0, intrinsic[1,1], intrinsic[1,2]],
            [0, 0, 1]]
        
        # load 2D colour frames and poses

        frame_ids = os.listdir(os.path.join(scene_dir,'train'))
        
        frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
        frame_ids = sorted(frame_ids)
        frames_file_list = []
        poses = read_pose(pose_path)
        for i, frame_id in tqdm(enumerate(frame_ids)):
            pose = poses[i]
            # skip frames with no valid pose
            if not np.all(np.isfinite(pose)):
                continue

            frame = {'file_name_image': os.path.join(scene_dir,'train', '%06d.png'%frame_id),
                    'file_name_label': 
                        os.path.join(scene_dir, 'semantic', '%06d.png'%frame_id),
                    'intrinsics': intrinsics,
                    'pose': pose,
                    }

            frames_file_list.append(frame)

        valid_data_num = len(frames_file_list)
        self.valid_data_num = valid_data_num
        total_ids = range(valid_data_num)
        train_ids = total_ids
        self.train_ids = train_ids
        self.train_num = len(train_ids)

        self.train_samples = {'image': [], 
                              'semantic_raw': [],
                              'semantic': [],
                              'T_wc': []}

        for idx in train_ids:
            image = cv2.imread(frames_file_list[idx]["file_name_image"])[:,:,::-1] # change from BGR uinit 8 
            image = image/255.0
            

            semantic = cv2.imread(frames_file_list[idx]["file_name_label"], cv2.IMREAD_UNCHANGED)


            T_wc = frames_file_list[idx]["pose"].reshape((4, 4))

            self.train_samples["image"].append(image)
            self.train_samples["semantic_raw"].append(semantic)
            self.train_samples["T_wc"].append(T_wc)

        self.intrinsics = np.asarray(intrinsics)


        for key in self.train_samples.keys():  # transform list of np array to array with batch dimension
            self.train_samples[key] = np.asarray(self.train_samples[key])

        train_semantic = self.train_samples["semantic_raw"]
        self.train_samples["semantic"] = train_semantic

        self.semantic_classes = np.unique((np.unique(self.train_samples["semantic"]))).astype(np.uint8)
        
        self.num_semantic_class = self.semantic_classes.shape[0]  # number of semantic classes
        self.mask_ids = np.ones(self.train_num)  # init self.mask_ids as full ones


        # remap existing semantic class labels to continuous label ranging from 0 to num_class-1
        self.train_samples["semantic_clean"] = self.train_samples["semantic"].copy()
        self.train_samples["semantic_remap"] = self.train_samples["semantic"].copy()
        self.train_samples["semantic_remap_clean"] = self.train_samples["semantic_clean"].copy()


        for i in range(self.num_semantic_class):
            self.train_samples["semantic_remap"][self.train_samples["semantic"]== self.semantic_classes[i]] = i
            self.train_samples["semantic_remap_clean"][self.train_samples["semantic_clean"]== self.semantic_classes[i]] = i


        self.train_samples["semantic_remap"] = self.train_samples["semantic_remap"].astype(np.uint8)
        self.train_samples["semantic_remap_clean"] = self.train_samples["semantic_remap_clean"].astype(np.uint8)

        print()
        print("Training Sample Summary:")
        for key in self.train_samples.keys(): 
            print("{} has shape of {}, type {}.".format(key, self.train_samples[key].shape, self.train_samples[key].dtype))
            
    def add_pixel_wise_noise_label(self, 
        sparse_views=False, sparse_ratio=0.0, random_sample=False, 
        noise_ratio=0.0, visualise_save=False, load_saved=False):
        """
        sparse_views: whether we sample a subset of dense semantic labels for training
        sparse_ratio: the ratio of frames to be removed/skipped if sampling a subset of labels
        random_sample: whether to random sample frames or interleavely/evenly sample, True--random sample; False--interleavely sample
        noise_ratio: the ratio of num pixels per-frame to be randomly perturbed
        visualise_save: whether to save the noisy labels into harddrive for later usage
        load_saved: use trained noisy labels for training to ensure consistency betwwen experiments
        """

        if not load_saved:
            num_pixel = self.img_h * self.img_w
            num_pixel_noisy = int(num_pixel*noise_ratio)
            train_sem = self.train_samples["semantic_remap"]

            for i in range(self.train_num):
                noisy_index_1d = np.random.permutation(num_pixel)[:num_pixel_noisy]
                faltten_sem = train_sem[i].flatten()

                faltten_sem[noisy_index_1d] = np.random.choice(self.num_semantic_class, num_pixel_noisy)
                # we replace the label of randomly selected num_pixel_noisy pixels to random labels from [1, self.num_semantic_class], 0 class is the none class
                train_sem[i] = faltten_sem.reshape(self.img_h, self.img_w)

            print("semantic labels are added noise {} percent area ratio.".format(noise_ratio))

            if visualise_save:
                noisy_sem_dir = os.path.join(self.semantic_class_dir, "noisy_pixel_sems_sr{}_nr{}".format(sparse_ratio, noise_ratio))
                if not os.path.exists(noisy_sem_dir):
                    os.makedirs(noisy_sem_dir)

                vis_noisy_semantic_list = []
                vis_semantic_clean_list = []

                colour_map_np = kitti_color_code
                # 101 classes in total from Replica, select the existing class from total colour map

                semantic_remap = self.train_samples["semantic_remap"] # [H, W, 3]
                semantic_remap_clean = self.train_samples["semantic_remap_clean"] # [H, W, 3]

                # save semantic labels
                for i in tqdm(range(self.train_num)):
                    vis_noisy_semantic = colour_map_np[semantic_remap[i]] # [H, W, 3]
                    vis_semantic_clean = colour_map_np[semantic_remap_clean[i]] # [H, W, 3]

                    imageio.imwrite(os.path.join(noisy_sem_dir, "semantic_class_{}.png".format(i)), semantic_remap[i])
                    imageio.imwrite(os.path.join(noisy_sem_dir, "vis_sem_class_{}.png".format(i)), vis_noisy_semantic)

                    vis_noisy_semantic_list.append(vis_noisy_semantic)
                    vis_semantic_clean_list.append(vis_semantic_clean)

        else:
            print("Load saved noisy labels.")
            noisy_sem_dir = os.path.join(self.semantic_class_dir, "noisy_pixel_sems_sr{}_nr{}".format(sparse_ratio, noise_ratio))
            assert os.path.exists(noisy_sem_dir)
            self.mask_ids = np.load(os.path.join(noisy_sem_dir, "mask_ids.npy"))
            semantic_img_list = []
            semantic_path_list = sorted(glob.glob(noisy_sem_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
            assert len(semantic_path_list)>0
            for idx in range(len(self.mask_ids)):
                semantic = imread(semantic_path_list[idx])
                semantic_img_list.append(semantic)
            self.train_samples["semantic_remap"]  = np.asarray(semantic_img_list)
            
    def sample_label_maps(self, sparse_ratio=0.5, K=0, random_sample=False, load_saved=False):
        """
        sparse_ratio means the ratio of removed training images, e.g., 0.3 means 30% of semantic labels are removed
        Input:
            sparse_ratio: the percentage of semantic label frames to be *removed*
            K: the number of frames to be removed, if this is speficied we override the results computed from sparse_ratio
            random_sample: whether to random sample frames or interleavely/evenly sample, True--random sample; False--interleavely sample
            load_saved: use pre-computed mask_ids from previous experiments
        """
        if load_saved is False:
            if K==0:
                K = int(self.train_num*sparse_ratio)  # number of skipped training frames, mask=0

            N = self.train_num-K  # number of used training frames,  mask=1

            if K==0: # in case sparse_ratio==0:
                return 
        
            if random_sample:
                self.mask_ids[:K] = 0
                np.random.shuffle(self.mask_ids)
            else:  # sample interleave
                if sparse_ratio<=0.5: # skip less/equal than half frames
                    assert K <= self.train_num/2
                    q, r = divmod(self.train_num, K)
                    indices = [q*i + min(i, r) for i in range(K)]
                    self.mask_ids[indices] = 0

                else: # skip more than half frames
                    assert K > self.train_num/2
                    self.mask_ids = np.zeros_like(self.mask_ids)  # disable all images and  evenly enable N images in total
                    q, r = divmod(self.train_num, N)
                    indices = [q*i + min(i, r) for i in range(N)]
                    self.mask_ids[indices] = 1

            print("sparse ratio: {}.".format(sparse_ratio))
        elif load_saved is True:
            noisy_sem_dir = os.path.join(self.semantic_class_dir, "noisy_pixel_sems_sr{}".format(sparse_ratio))
            self.mask_ids = np.load(os.path.join(noisy_sem_dir, "mask_ids.npy"))