import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio 
import json
import cv2


class Dataset(data.Dataset): 
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene

        if 'precrop_epoch' in kwargs:
            self.precrop = True
            self.precrop_epoch = kwargs['precrop_epoch']
            self.precrop_frac = kwargs['precrop_frac']
        else:
            self.precrop = False

        self.input_ratio = kwargs['input_ratio']


        self.epoch = 0
        self.data_root = os.path.join(data_root, scene)
        self.scene = scene
        self.split = split

        self.c2ws = []
        self.ixts = []
        self.image_paths = []

        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format(self.split))))
        b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        scene_info = {'ixts': [], 'exts': [], 'img_paths': []}

        ixt = np.eye(3)
        ixt[0][2], ixt[1][2] = 400., 400.
        focal = .5 * 800 / np.tan(.5 * json_info['camera_angle_x'])
        ixt[0][0], ixt[1][1] = focal, focal
        ixt = ixt.astype(np.float32)

        for frame in json_info['frames']:
            self.image_paths.append(os.path.join(self.data_root, frame['file_path'][2:] + '.png'))
            self.c2ws.append((np.array(frame['transform_matrix']) @ b2c).astype(np.float32))
            self.ixts.append(ixt)
        if 'bbox' in cfg:
            self.bounds = np.array(cfg.bbox).reshape((2, 3))

        cams = kwargs['cams']
        b,e,s = cams
        cam_len = len(self.image_paths)

        e = cam_len if e == -1 else e

        self.image_paths = self.image_paths[b:e:s]
        self.c2ws = self.c2ws[b:e:s]
        self.ixts = self.ixts[b:e:s]


        self.batch_size = cfg.task_arg.N_rays

    def __getitem__(self, index):
        img_path, c2w, ixt = self.image_paths[index], self.c2ws[index], self.ixts[index]
        img = (np.array(imageio.imread(img_path)) / 255.).astype(np.float32)
        if img.shape[2] == 4:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            # images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            # img, mask = img[..., :3], img[..., 3]
            # img[mask!=1.] = 1.

        if self.input_ratio != 1.:
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
            ixt = ixt.copy()
            ixt[:2] *= self.input_ratio

        if self.precrop and self.epoch < self.precrop_epoch and self.precrop_frac > 0.:
            h, w = img.shape[:2]
            crop_h, crop_w = int(h//2 * self.precrop_frac), int(w//2 * self.precrop_frac)
            img = img[crop_h:-crop_h, crop_w:-crop_w]
            ixt[0, 2] -= crop_w
            ixt[1, 2] -= crop_h

        near_far = np.array([2., 6.]).astype(np.float32)

        rays, rays_rgb = self.gen_rays(c2w, ixt, img, near_far)
        
        if self.split == 'train':
            ids = np.random.choice(len(rays), self.batch_size, replace=False)
            rays = rays[ids]
            rays_rgb = rays_rgb[ids]
        ret = {'rays': rays, 'rgb': rays_rgb}
        ret.update({'ixt': ixt, 'c2w': c2w})
        ret.update({'meta': {'img_path': img_path, 'h': img.shape[0], 'w': img.shape[1]}})
        return ret

    def __len__(self):
        return len(self.image_paths)

    def gen_rays(self, c2w, ixt, img, near_far):
        H, W = img.shape[:2]
        rays_o = c2w[:3, 3]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        X, Y = X.astype(np.float32), Y.astype(np.float32)
        XYZ = np.stack([(X-ixt[0][2])/ixt[0][0], (Y-ixt[1][2])/ixt[1][1], np.ones_like(X)], axis=-1)
        rays_d = (XYZ[..., None, :] * c2w[:3, :3]).sum(-1)
        rays_d = rays_d.reshape(-1, 3)

        rays_rgb = img.reshape(-1, 3)
        rays = np.concatenate((rays_o[None].repeat(H*W, axis=0), rays_d, near_far[None].repeat(H*W, axis=0)), axis=-1)
        return rays.astype(np.float32), rays_rgb
