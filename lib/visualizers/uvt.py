import matplotlib.pyplot as plt
from lib.utils import data_utils
from lib.utils import img_utils
from lib.config import cfg
import numpy as np
import torch.nn.functional as F
import torch
import imageio
import os
import cv2

class Visualizer:
    def __init__(self,):
        self.write_video = cfg.write_video
        self.imgs = []
        os.system('mkdir -p {}'.format(cfg.result_dir))
        os.system('mkdir -p {}'.format(cfg.result_dir + '/imgs'))
        os.system('mkdir -p {}'.format(cfg.result_dir + '/imgs_time'))

    def visualize(self, output, batch):
        h, w = batch['meta']['h'].item(), batch['meta']['w'].item()
        idx = batch['meta']['idx'].item()
        img = output['rgb'].reshape(h, w, 3).detach().cpu().numpy()
        imageio.imwrite(os.path.join(cfg.result_dir, 'imgs/{:06d}.png'.format(idx)), img)
        img = (img * 255.).astype(np.uint8)
        year, month, day = batch['meta']['time'][0][0].detach().cpu().numpy().tolist()
        img = cv2.putText(img, '{:04d}-{:02d}-{:02d}'.format(year, month, day),(w//2-100, h//2-100),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        imageio.imwrite(os.path.join(cfg.result_dir, 'imgs_time/{:06d}.png'.format(idx)), img)
        self.imgs.append(img)

    def summarize(self):
        imageio.mimwrite(os.path.join(cfg.result_dir, 'color.mp4'), self.imgs, fps=cfg.fps)



