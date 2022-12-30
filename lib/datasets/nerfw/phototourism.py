from dataclasses import replace
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from torchvision import transforms as T
from lib.config import cfg

from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary 


"""
self.scene_name = brandenburg
N_images = 773
self.files : [773 rows x 4 columns]  filename(xxxxxx.jpg)  id  split  dataset
self.image_to_cam : dict(1363 pairs) 所有的匹配对 给定img_id 可以去搜需要的image
self.img_ids : dict(773 pairs) 
self.image_paths : dict(773 pairs) key:id value: xxxxx.jpg
self.Ks : dict(773, 每一张的内参矩阵K)
self.poses : <class 'numpy.ndarray'> (773, 3, 4)
self.poses_dict : 按照id去match一下对应的pose
    self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)} # match 通过img_ids 将每一张image 和pose 对应一下 
    len(self.poses_dict) = 773
self.xyz_world : (100040, 3) [-0.04926536 -0.01590442  0.74996179]
self.nears : dict(773) key:id value: 对应的nears
self.fars : dict(773) key:id value: 对应的fars
self.img_ids_train : len() = 763 通过在files里面查split是否是train来选择id, list(763) 存id
self.N_images_train = 763
self.img_ids_test : len() = 10 通过在files里面查split是否是test来选择id, list(763) 存id
self.N_images_test = 10
all_rays.shape # (8742109, 9)
all_rgbs.shape # (8742109, 3)
"""


class PhototourismDataset(Dataset):
    # def __init__(self, data_root, split='train', img_downscale=1, val_num=1, use_cache=False):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.use_cache = kwargs['use_cache']
        self.split = kwargs['split']
        self.data_root = kwargs['data_root']
        self.img_downscale = kwargs['img_downscale']
        assert self.img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        if self.split == 'val': # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)
        self.val_num = max(1, kwargs['val_num']) # at least 1
        self.use_cache = kwargs['use_cache']
        self.define_transforms()

        self.read_meta()
        self.white_back = False
        
        self.batch_size = cfg.task_arg.N_rays

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.data_root, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()] # remove data without id
        self.files.reset_index(inplace=True, drop=True)


        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.data_root, f'cache/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            ################################## new ##################################
            with open(os.path.join(self.data_root, f'cache/img_to_cam_id.pkl'), 'rb') as f:
                self.image_to_cam = pickle.load(f)
            ################################## new ##################################
            with open(os.path.join(self.data_root, f'cache/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = read_images_binary(os.path.join(self.data_root, 'dense/sparse/images.bin'))
            img_path_to_id = {}

            ############################################# ++ #######################################
            self.image_to_cam = {}
            # id之间的对应关系
            ############################################# ++ #######################################

            for v in imdata.values():
                img_path_to_id[v.name] = v.id
                ############################################# ++ #######################################
                self.image_to_cam[v.id] = v.camera_id
                ############################################# ++ #######################################
            self.img_ids = []
            self.image_paths = {} # {id: filename}
            for filename in list(self.files['filename']):
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        
        if self.use_cache:
            with open(os.path.join(self.data_root, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
                self.Ks = pickle.load(f) 
                
        else:
            self.Ks = {} # {id: K}
            camdata = read_cameras_binary(os.path.join(self.data_root, 'dense/sparse/cameras.bin'))

            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam_id = self.image_to_cam[id_]
                cam = camdata[cam_id]
                img_w, img_h = int(cam.params[2]*2), int(cam.params[3]*2)
                img_w_, img_h_ = img_w//self.img_downscale, img_h//self.img_downscale
                K[0, 0] = cam.params[0]*img_w_/img_w # fx
                K[1, 1] = cam.params[1]*img_h_/img_h # fy
                K[0, 2] = cam.params[2]*img_w_/img_w # cx
                K[1, 2] = cam.params[3]*img_h_/img_h # cy
                K[2, 2] = 1
                self.Ks[cam_id] = K 

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.data_root, 'cache/poses.npy'))
            # print(type(self.poses), self.poses.shape) # <class 'numpy.ndarray'> (773, 3, 4)
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.data_root, 'cache/xyz_world.npy'))
            with open(os.path.join(self.data_root, f'cache/nears.pkl'), 'rb') as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.data_root, f'cache/fars.pkl'), 'rb') as f:
                self.fars = pickle.load(f)
            # print(self.xyz_world.shape, self.xyz_world[0]) # (100040, 3) [-0.04926536 -0.01590442  0.74996179]
            # print(self.nears.keys()) # # dict(773 keys)


        else: 
            pts3d = read_points3d_binary(os.path.join(self.data_root, 'dense/sparse/points3D.bin'))
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {} # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()
            scale_factor = max_far/5 # so that the max far is scaled to 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor 
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor

        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)} # match 通过img_ids 将每一张image 和pose 对应一下 

            
        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids) 
                                    if self.files.loc[i, 'split']=='train']

        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='test']

        self.N_images_train = len(self.img_ids_train) 
        self.N_images_test = len(self.img_ids_test) 

        ############# self : 跑 run.py 时候的idea ########################
        # self.poses_train = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids_train)}
        # self.poses_test = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids_test)}
        ############# self : 跑 run.py 时候的idea ########################

        if self.split == 'train': # create buffer of all rays and rgb data
            if self.use_cache:
                all_rays = np.load(os.path.join(self.data_root,
                                                f'cache/rays{self.img_downscale}.npy'))
                self.all_rays = torch.from_numpy(all_rays)
                all_rgbs = np.load(os.path.join(self.data_root,
                                                f'cache/rgbs{self.img_downscale}.npy'))
                self.all_rgbs = torch.from_numpy(all_rgbs)
            else:
                self.all_rays = []
                self.all_rgbs = []
                for id_ in self.img_ids_train:
                    c2w = torch.FloatTensor(self.poses_dict[id_])

                    img = Image.open(os.path.join(self.data_root, 'dense/images',
                                                  self.image_paths[id_])).convert('RGB')
                    img_w, img_h = img.size
                    if self.img_downscale > 1:
                        img_w = img_w//self.img_downscale
                        img_h = img_h//self.img_downscale
                        img = img.resize((img_w, img_h), Image.LANCZOS)
                    img = self.transform(img) # (3, h, w)
                    img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                    self.all_rgbs += [img]

                    ################## original #######################
                    # directions = get_ray_directions(img_h, img_w, self.Ks[id_])
                    ################## original #######################

                    ################## new #######################
                    directions = get_ray_directions(img_h, img_w, self.Ks[self.image_to_cam[id_]])
                    ################## new #######################


                    rays_o, rays_d = get_rays(directions, c2w)
                    rays_t = id_ * torch.ones(len(rays_o), 1)

                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                                                self.fars[id_]*torch.ones_like(rays_o[:, :1]),
                                                rays_t],
                                                1)] # (h*w, 9)
                # import ipdb; ipdb.set_trace()
                                    
                self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
                self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        
        elif self.split in ['val', 'test_train']: # use the first image as val image (also in train)
            self.val_id = self.img_ids_train[0]

        else: # for testing, create a parametric rendering path
            # test poses and appearance index are defined in eval.py
            pass

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
            # return len(self.img_ids_train)
        if self.split == 'test':
            return len(self.img_ids_test)
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        sample = {}
        if self.split == 'train':
            sample['rays'] = self.all_rays[idx, :8]
            sample['ts'] = self.all_rays[idx, 8].long()
            sample['rgb'] = self.all_rgbs[idx]
        else:
            # id_ = self.val_id
            # import ipdb; ipdb.set_trace() 
            id_ = self.img_ids_test[idx]
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[id_])

            img = Image.open(os.path.join(self.data_root, 'dense/images',
                                            self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgb'] = img
            directions = get_ray_directions(img_h, img_w, self.Ks[self.image_to_cam[id_]])
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d,
                                self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                                self.fars[id_]*torch.ones_like(rays_o[:, :1])],
                                1) # (h*w, 8)
            sample['rays'] = rays
            sample['ts'] = (id_ * torch.ones(len(rays), dtype=torch.long))
        # if self.split == 'train':
        #     if len(rays) <= self.batch_size:
        #         ids = np.random.choice(len(rays), len(rays-1), replace=False)
        #     else:
        #         ids = np.random.choice(len(rays), self.batch_size, replace=False)
        #     rays = rays[ids]
        #     sample['rgb'] = img[ids]

            sample['meta'] = {}
            sample['meta']['img_path_id'] = {'id': id_, 'path': self.image_paths[id_]}
            sample['meta']['h'] = img_h
            sample['meta']['w'] = img_w

        return sample