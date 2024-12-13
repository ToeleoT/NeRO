from dataset import kitti_Dataset
import numpy as np
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import random
from network import *
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import imageio
import warnings
warnings.filterwarnings("ignore")
from tools import *
from points_range import *



import argparse

parser = argparse.ArgumentParser(description='argparse learning')
parser.add_argument('--data_dir', type=str, default='/clever/volumes/gfs-35-42/wrb/road_method_code', help='data dirctor')
parser.add_argument('--save_dir', type=str, default='/clever/volumes/gfs-35-42/wrb/road_method_code', help='data dirctor')
parser.add_argument('--pose_dir', type=str, default='00.txt', help='pose dirctory')
parser.add_argument('--sem', type=bool, default=True, help='open semantic or not')
parser.add_argument('--pe_mod', type=int, default=1, help='which pe method')
parser.add_argument('--global_steps', type=int, default=5000, help='total training epoch')
parser.add_argument('--data_type', type=str, default="lidar", help='which dataset want to use')
parser.add_argument('--denoise', type=int, default=0, help='denoise or not')
parser.add_argument('--denoise_ratio', type=float, default=0.5, help='denoise ratio')
parser.add_argument('--sparse_label', type=int, default=0, help='sparse label or not')
parser.add_argument('--sparse_ratio', type=float, default=0.1, help='sparse label ratio')
args = parser.parse_args()


kitti_colour_code = np.array([(128,64,128),(244,35,232),(153,153,153),(111,74,0)]).astype(np.uint8)

scene_dir = args.data_dir
save_dir = args.save_dir


if args.data_type == 'lidar':
    points_create_camera = points_range('camera')
    max_val_in_c,min_val_in_c = points_create_camera.get_min_max()
    points_create_lidar = points_range('lidar')
    max_val_in_l,min_val_in_l = points_create_lidar.get_min_max()


    max_val_in = []
    for c,l in zip(max_val_in_c,max_val_in_l):
        if c >= l:
            max_val_in.append(c + 0.01)
        else:
            max_val_in.append(l + 0.01)
    min_val_in = []
    for c,l in zip(min_val_in_c,min_val_in_l):
        if c >= l:
            min_val_in.append(l - 0.01)
        else:
            min_val_in.append(c - 0.01)
            
elif args.data_type == 'sfm_dense':
    points_create_camera = points_range('camera')
    max_val_in_c,min_val_in_c = points_create_camera.get_min_max()
    points_create_sfm = points_range('sfm_dense')
    max_val_in_l,min_val_in_l = points_create_sfm.get_min_max()


    max_val_in = []
    for c,l in zip(max_val_in_c,max_val_in_l):
        if c >= l:
            max_val_in.append(c + 0.01)
        else:
            max_val_in.append(l + 0.01)
    min_val_in = []
    for c,l in zip(min_val_in_c,min_val_in_l):
        if c >= l:
            min_val_in.append(l - 0.01)
        else:
            min_val_in.append(c - 0.01)
elif args.data_type == 'sfm_sparse':
    points_create_camera = points_range('camera')
    max_val_in_c,min_val_in_c = points_create_camera.get_min_max()
    points_create_sfm = points_range('sfm_sparse')
    max_val_in_l,min_val_in_l = points_create_sfm.get_min_max()


    max_val_in = []
    for c,l in zip(max_val_in_c,max_val_in_l):
        if c >= l:
            max_val_in.append(c + 0.01)
        else:
            max_val_in.append(l + 0.01)
    min_val_in = []
    for c,l in zip(min_val_in_c,min_val_in_l):
        if c >= l:
            min_val_in.append(l - 0.01)
        else:
            min_val_in.append(c - 0.01)
            
elif args.data_type == 'camera':
    points_create_camera = points_range('camera')
    max_val_in_c,min_val_in_c = points_create_camera.get_min_max()


    max_val_in = []
    for c in max_val_in_c:
        max_val_in.append(c + 0.01)
    min_val_in = []
    for c in min_val_in_c:
        min_val_in.append(c - 0.01)

sem = args.sem

Dataset_input = kitti_Dataset(scene_dir,args.pose_dir)
if sem:
    train_semantic_true = Dataset_input.train_samples["semantic_remap"]
    train_semantic = Dataset_input.train_samples["semantic_remap"]
    if args.denoise == 1:
        Dataset_input.add_pixel_wise_noise_label(noise_ratio=args.denoise_ratio, visualise_save=True)
    if args.sparse_label == 1:
        Dataset_input.sample_label_maps(sparse_ratio=args.sparse_ratio)
        sparse_id = Dataset_input.mask_ids
c2w = Dataset_input.train_samples['T_wc']
intrinsics = torch.from_numpy(Dataset_input.intrinsics).float().cuda()

train_image = Dataset_input.train_samples["image"]
img_h = train_image[0].shape[0]
img_w = train_image[0].shape[1]
semantic_classes = Dataset_input.num_semantic_class
to8b_np = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

global_step = args.global_steps
skips = [2]
input_ch  = 2
model = NeRO(num_semantic_classes=semantic_classes,
             D=2, W=64,
             input_ch=input_ch, skips=skips,max_val = max_val_in,min_val = min_val_in,mod = args.pe_mod,sem = sem).cuda()
optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4)
logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
CrossEntropyLoss = nn.CrossEntropyLoss()
crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label)
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
SmoothL1Loss = torch.nn.SmoothL1Loss()
wgt_sem_loss = float(4e-2)


point_path = np.sort(glob.glob(os.path.join(scene_dir,'lidar','*.npy')))
point_path_grid = np.sort(glob.glob(os.path.join(scene_dir,'camera_offset','*.npy')))
length = len(point_path_grid)
def train(point_path,point_path_grid,global_step,length,img_h,img_w):
    avg_psnr = []
    avg_miou = []
    avg_acc = []
    avg_cls_acc = []
    for step in tqdm(range(0,global_step+1)):

        ind_shuffle = random.sample(range(length), 1)[0]
        all_points = torch.from_numpy(np.load(point_path[ind_shuffle])).float().cuda()
        points = all_points[:,:2]
        z_true = all_points[:,-1]
        images = torch.from_numpy(train_image[ind_shuffle]).cuda()
        if sem:
            semantics = torch.from_numpy(train_semantic[ind_shuffle]).cuda()
            semantics_miou = torch.from_numpy(train_semantic_true[ind_shuffle]).cuda()
        c2w_out = torch.from_numpy(c2w[ind_shuffle]).float().cuda()

        optimizer.zero_grad()

        inputs = points
        outputs = model(inputs)

        z_cal = outputs[:,0]
        z_loss = img2mse(z_cal,z_true)


        all_points_another = torch.from_numpy(np.load(point_path_grid[ind_shuffle])).float().cuda()
        points_in = all_points_another[:,:2]
        outputs_another = model(points_in)
        z = outputs_another[:,0]
        rgb_out = outputs_another[:,1:4]
        if sem:
            semantic_logit = outputs_another[:,4:]

        world_point_out = torch.cat((points_in,z.unsqueeze(1)),axis = 1)
        world_point_out = torch.cat((world_point_out,torch.tensor(1.).cuda().repeat(world_point_out.shape[0]).unsqueeze(dim =1)),dim = 1)

        chaiss2cam = torch.linalg.inv(c2w_out) @ torch.linalg.inv(torch.from_numpy(camera2chassis)).cuda()
        camera_point = torch.mm(world_point_out,chaiss2cam[:3,...].T)
        camera_point = camera_point.detach().cpu().numpy()

        camera_point_index = np.argwhere(camera_point[..., 2] > 0.0).flatten()

        rgb_out = rgb_out[camera_point_index]

        if sem:
            semantic_logit = semantic_logit[camera_point_index]

        camera_point = torch.from_numpy(camera_point[camera_point_index]).cuda()

        u = camera_point[..., 0] / camera_point[..., 2] * intrinsics[0, 0] + intrinsics[0, 2]
        v = camera_point[..., 1] / camera_point[..., 2] * intrinsics[1, 1] + intrinsics[1, 2]
        u_v = torch.cat((u.unsqueeze(1),v.unsqueeze(1)),axis = 1)
        u_v = u_v.int().cpu().numpy()
        images_np = images.cpu().numpy()

        index_need = np.argwhere(((u_v[:,0] >= 0) & (u_v[:,0] < img_w)) & ((u_v[:,1] >= 0) & (u_v[:,1] < img_h))).flatten()
        u_v = u_v[index_need]
        rgb_out = rgb_out[index_need]
        if sem:
            semantic_logit = semantic_logit[index_need]
        images_np8 = to8b_np(images_np)
        no_black = np.column_stack(np.where(np.any(images_np8 != [0, 0, 0], axis=-1)))

        no_black_index = np.flatnonzero(np.in1d(asvoid(u_v[:,::-1].astype('int')), asvoid(no_black.astype('int'))))

        if len(no_black_index) == 0:
            img_loss = torch.tensor(0.).cuda()
            psnr = mse2psnr(torch.tensor(0.))
            sem_loss = torch.tensor(0.).cuda()
            color_with_semantic_loss = torch.tensor(0.).cuda()
        else:
            rgb_out_predict = rgb_out[no_black_index]
            if sem:
                semantic_logit_predict = semantic_logit[no_black_index]
            images_true = images[tuple(np.transpose(u_v[:,::-1][no_black_index]))]

            if sem:
                semantics_true = semantics[tuple(np.transpose(u_v[:,::-1][no_black_index]))]
                semantics_miou_true = semantics_miou[tuple(np.transpose(u_v[:,::-1][no_black_index]))]
            img_loss = img2mse(rgb_out_predict,images_true)
            psnr = mse2psnr(torch.tensor(img_loss.item()))
            if sem:
                if args.sparse_label == 1:
                    if sparse_id[ind_shuffle] == 1:
                        sem_loss = crossentropy_loss(semantic_logit_predict,semantics_true.long())
                    else:
                        sem_loss = torch.tensor(0).cuda()
                else:
                    sem_loss = crossentropy_loss(semantic_logit_predict,semantics_true.long())
                color_with_semantic_loss = img_loss + sem_loss
            else:
                color_with_semantic_loss = img_loss
        total_loss = z_loss + color_with_semantic_loss
        total_loss.backward()
        optimizer.step()
        tqdm.write(f"[TRAIN] Iter: {step} "
                           f"Total Loss: {total_loss.item()} "
                           f"Z Loss: {z_loss.item()} "
                           f"PSNR: {psnr.item()} "
                           f"Semantic Loss: {sem_loss.item()} "
                           f"Color Loss: {img_loss.item()} ")
        
        decay_rate = 0.1
        decay_steps = 250e3
        new_lrate = 5e-4 * (decay_rate ** (step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
            
    
#         if step % 5000 == 0:
#             ckpt_dir = os.path.join(save_dir, "checkpoints")
#             if not os.path.exists(ckpt_dir):
#                 os.makedirs(ckpt_dir)

#             ckpt_file = os.path.join(ckpt_dir, '{:06d}.pth'.format(step))
#             torch.save({
#                 'global_step': step,
#                 'network_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#             }, ckpt_file)
#             print('Saved checkpoints at', ckpt_file)
def evaluate(grid_point_path,img_h,img_w):
    result_c = []
    result_s = []
    model.eval()
    
    length = len(grid_point_path)
    with torch.no_grad():
    #     avg_psnr = []
        avg_miou = []
        avg_acc = []
        avg_cls_acc = []
        for ind_shuffle in tqdm(range(length)):
            all_points = torch.from_numpy(np.load(grid_point_path[ind_shuffle])).float().cuda()
            images = torch.from_numpy(train_image[ind_shuffle]).cuda()
            if sem:
                semantics = torch.from_numpy(train_semantic[ind_shuffle]).cuda()
                semantics_miou = torch.from_numpy(train_semantic_true[ind_shuffle]).cuda()
            points = all_points[:,:2]
            z_true = all_points[:,-1]
            c2w_out = torch.from_numpy(c2w[ind_shuffle]).float().cuda()
            inputs = points
            outputs = model(inputs)
            z = outputs[:,0]
            rgb_out = outputs[:,1:4]
            if sem:
                semantic_logit = outputs[:,4:]
            world_point_out = torch.cat((points,z.unsqueeze(1)),axis = 1)
            world_point_output = world_point_out
            world_point_out = torch.cat((world_point_out,torch.tensor(1.).cuda().repeat(world_point_out.shape[0]).unsqueeze(dim =1)),dim = 1)

            chaiss2cam = torch.linalg.inv(c2w_out) @ torch.linalg.inv(torch.from_numpy(camera2chassis)).cuda()
            camera_point = torch.mm(world_point_out,chaiss2cam[:3,...].T)
            camera_point = camera_point.detach().cpu().numpy()
            camera_point_index = np.argwhere(camera_point[..., 2] > 0.0).flatten()
            rgb_out = rgb_out[camera_point_index]
            if sem:
                semantic_logit = semantic_logit[camera_point_index]
            world_point_output = world_point_output[camera_point_index]
            camera_point = torch.from_numpy(camera_point[camera_point_index]).cuda()
            u = camera_point[..., 0] / camera_point[..., 2] * intrinsics[0, 0] + intrinsics[0, 2]
            v = camera_point[..., 1] / camera_point[..., 2] * intrinsics[1, 1] + intrinsics[1, 2]
            u_v = torch.cat((u.unsqueeze(1),v.unsqueeze(1)),axis = 1)
            u_v = u_v.int().cpu().numpy()
            images_np = images.cpu().numpy()
            index_need = np.argwhere(((u_v[:,0] >= 0) & (u_v[:,0] < img_w)) & ((u_v[:,1] >= 0) & (u_v[:,1] < img_h))).flatten()
            u_v = u_v[index_need]
            rgb_out = rgb_out[index_need]
            if sem:
                semantic_logit = semantic_logit[index_need]
            world_point_output = world_point_output[index_need]
            images_np8 = to8b_np(images_np)
            no_black = np.column_stack(np.where(np.any(images_np8 != [0, 0, 0], axis=-1)))
            no_black_index = np.flatnonzero(np.in1d(asvoid(u_v[:,::-1].astype('int')), asvoid(no_black.astype('int'))))
            rgb_out_predict = rgb_out[no_black_index]
            world_point_output = world_point_output[no_black_index]
            if sem:
                semantic_logit_predict = semantic_logit[no_black_index]
                semantics_true = semantics[tuple(np.transpose(u_v[:,::-1][no_black_index]))]
                semantics_miou_true = semantics_miou[tuple(np.transpose(u_v[:,::-1][no_black_index]))]

            exp_dir = os.path.join(save_dir, "experiment_hash")
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            world_points_rgb = torch.cat((world_point_output,rgb_out_predict),axis = 1)
            if sem:
                semantic_logit = logits_2_label(semantic_logit_predict)
                miou, miou_valid_class, total_accuracy, class_average_accuracy, ious = calculate_segmentation_metrics(semantics_miou_true.long().cpu().numpy(),semantic_logit.cpu().numpy(),3)
                avg_miou.append(miou.item())
                avg_acc.append(total_accuracy.item())
                avg_cls_acc.append(class_average_accuracy.item())
                world_points_sem = torch.cat((world_point_output,torch.unsqueeze(semantic_logit,dim = 1)),axis = 1)
                world_points_sem = world_points_sem.detach().cpu().numpy()
            world_points_array = world_points_rgb.detach().cpu().numpy()
            
            for entry in world_points_array:
                temp = [entry[0],entry[1],entry[2],to8b_np(entry[3]),to8b_np(entry[4]),to8b_np(entry[5])]
                result_c.append(temp)
            if sem:
                for entry in world_points_sem:
                    sem_color = kitti_colour_code[int(entry[3])]
                    temp = [entry[0],entry[1],entry[2],sem_color[0],sem_color[1],sem_color[2]]
                    result_s.append(temp)

        
    canvas_color = draw_lidar(np.array(result_c))
    imageio.imwrite('color.png', canvas_color)
    canvas_sem = draw_lidar(np.array(result_s))
    imageio.imwrite('semantic.png', canvas_sem)
     
train(point_path,point_path_grid,global_step,length,img_h,img_w)
evaluate(point_path_grid,img_h,img_w)
