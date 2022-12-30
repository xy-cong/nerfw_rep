import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.nerf.mlp import NeRF
from lib.networks.encoding import get_encoder
from lib.networks.nerf import utils
from lib.config import cfg

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        net_cfg = cfg.network
        self.cascade_samples = cfg.task_arg.cascade_samples

        self.xyz_encoder, input_ch = get_encoder(net_cfg.xyz_encoder)
        self.dir_encoder, input_ch_views = get_encoder(net_cfg.dir_encoder)
        self.nerf = NeRF(input_ch, input_ch_views, net_cfg.nerf)

        if len(self.cascade_samples) > 1:
            self.cascade = True
            self.nerf_fine = NeRF(input_ch, input_ch_views, net_cfg.nerf)
        else:
            self.cascade = False

    def forward_network(self, xyz, xyz_dir, network):
        N_rays, N_samples = xyz.shape[:2]
        xyz, xyz_dir = xyz.reshape(-1, xyz.shape[-1]), xyz_dir.reshape(-1, xyz_dir.shape[-1])
        xyz_encoding = self.xyz_encoder(xyz)
        dir_encoding = self.dir_encoder(xyz_dir)
        net_output = network(torch.cat([xyz_encoding, dir_encoding], dim=-1))
        return net_output.reshape(N_rays, N_samples, -1)

    def render_rays(self, rays, batch):
        rays_o, rays_d, near, far  = rays[:, :3], rays[:, 3:6], rays[:, 6], rays[:, 7]
        viewdir = rays_d / rays_d.norm(dim=-1, keepdim=True)
        near = torch.clamp_min(near, 1e-8)
        z_vals = utils.sample_along_ray(near, far, self.cascade_samples[0])

        if self.training:
            z_vals = utils.perturb_samples(z_vals)
        xyz = rays_o[:, None] + rays_d[:, None] * z_vals[:, :, None]
        xyz_dir = viewdir[:, None].repeat(1, self.cascade_samples[0], 1)

        raw = self.forward_network(xyz, xyz_dir, self.nerf)
        ret = utils.raw2outputs(raw, z_vals, rays_d, cfg.task_arg.white_bkgd)

        if self.cascade:
            output = {}
            for key in ret:
                output[key + '_0'] = ret[key]
            weights = ret['weights'].clone().detach()
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = utils.sample_pdf(z_vals_mid, weights[..., 1:-1], self.cascade_samples[1], det=not self.training)
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            xyz = rays_o[:, None] + rays_d[:, None] * z_vals[:, :, None]
            xyz_dir = viewdir[:, None].repeat(1, self.cascade_samples[0] + self.cascade_samples[1], 1)

            raw = self.forward_network(xyz, xyz_dir, self.nerf_fine)
            ret = utils.raw2outputs(raw, z_vals, rays_d, cfg.task_arg.white_bkgd)

            for key in ret:
                output[key + '_1'] = ret[key]
            return output
        else:
            return ret


    def batchify_rays(self, rays, batch):
        all_ret = {}
        chunk = cfg.task_arg.chunk_size
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret

    def forward(self, batch):
        B, N_rays, C = batch['rays'].shape
        ret = self.batchify_rays(batch['rays'].reshape(-1, C), batch)
        return {k:ret[k].reshape(B, N_rays, -1) for k in ret}
