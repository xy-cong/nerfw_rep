
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.nerfw.mlp import NeRF 
from lib.networks.encoding import get_encoder
from lib.networks.nerfw import utils
from lib.config import cfg 

import commentjson as json
import tinycudann as tcnn

with open("tiny-cuda-nn/data/config_hash.json") as f:
	config = json.load(f)
 
"""
encoding = tcnn.Encoding(n_input_dims, config["encoding"])
network = tcnn.Network(encoding.n_output_dims, n_output_dims, config["network"])
model = torch.nn.Sequential(encoding, network)
"""

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        net_cfg = cfg.network
        self.cascade_samples = cfg.task_arg.cascade_samples
        xyz_n_input_dims = 3
        xyz_n_output_dims = 63
        dir_n_output_dims = 27
        
        encoding_xyz = tcnn.NetworkWithInputEncoding(
            xyz_n_input_dims, xyz_n_output_dims,
            config["encoding"], config["network"]
        )
        self.dir_encoder, input_ch_views = get_encoder(net_cfg.dir_encoder)
        self.xyz_encoder = encoding_xyz
        self.embeddings = {
            'xyz': self.xyz_encoder,
            'dir': self.dir_encoder
        }
        if net_cfg.nerf.encode_a:
            self.appearance_encoder = nn.Embedding(net_cfg.nerf.N_vocab, net_cfg.nerf.N_a) 
            self.embeddings['a'] = self.appearance_encoder
        self.nerf = NeRF(xyz_n_output_dims, dir_n_output_dims, 'coarse', net_cfg.nerf)

        if len(self.cascade_samples) > 1:
            self.cascade = True
            self.nerf_fine = NeRF(xyz_n_output_dims, dir_n_output_dims, 'fine', net_cfg.nerf)
        else:
            self.cascade = False
 
    def forward_network(self, xyz, xyz_dir, ts, network):
        N_rays, N_samples = xyz.shape[:2]
        net_output = []

        batch_chunk = cfg.task_arg.chunk_size # 1024

        for batch in range(0, xyz.shape[0], batch_chunk):
            xyz_batch, xyz_dir_batch = xyz[batch:batch+batch_chunk].reshape(-1, xyz[batch:batch+batch_chunk].shape[-1]), xyz_dir[batch:batch+batch_chunk].reshape(-1, xyz_dir[batch:batch+batch_chunk].shape[-1])
            xyz_encoding = self.embeddings['xyz'](xyz_batch)
            dir_encoding = self.embeddings['dir'](xyz_dir_batch)
            if cfg.network.nerf.encode_a:
                appearance_encoding = self.embeddings['a'](ts[batch:batch+batch_chunk])
                appearance_encoding = appearance_encoding.reshape(-1, appearance_encoding.shape[-1])
            net_output_chunk = network(torch.cat([xyz_encoding, dir_encoding, appearance_encoding], dim=-1))
            net_output.append(net_output_chunk)
            
       
        output = torch.cat(net_output, 0)
        return output.reshape(N_rays, N_samples, -1)
       
        # xyz_encoding = self.embeddings['xyz'](xyz)
        # dir_encoding = self.embeddings['dir'](xyz_dir)

        # if cfg.network.nerf.encode_a:
        #     appearance_encoding = self.embeddings['a'](ts)
        #     appearance_encoding = appearance_encoding.reshape(-1, appearance_encoding.shape[-1])
        # print("forword_network: xyz_encoding.shape, dir_encoding.shape,appearance_encoding.shape", xyz_encoding.shape, dir_encoding.shape,appearance_encoding.shape)
        # chunk = cfg.task_arg.chunk_size//8
        # net_output = []
        # # import ipdb; ipdb.set_trace()
        # for i in range(0, xyz_encoding.shape[0], chunk):
        #     net_output_chunk = network(torch.cat([xyz_encoding[i:i+chunk], dir_encoding[i:i+chunk], appearance_encoding[i:i+chunk]], dim=-1))
        #     net_output += net_output_chunk
        # output = torch.cat(net_output, 0)
        # return output.reshape(N_rays, N_samples, -1)

    def render_rays(self, rays, ts, batch):
        # import ipdb; ipdb.set_trace()
        rays_o, rays_d, near, far  = rays[:, :3], rays[:, 3:6], rays[:, 6], rays[:, 7]
        viewdir = rays_d / rays_d.norm(dim=-1, keepdim=True)
        near = torch.clamp_min(near, 1e-8)
        z_vals = utils.sample_along_ray(near, far, self.cascade_samples[0])
        if self.training:
            z_vals = utils.perturb_samples(z_vals)
        xyz = rays_o[:, None] + rays_d[:, None] * z_vals[:, :, None]
        ts_coarse = ts[:, None].repeat(1, self.cascade_samples[0])
        xyz_dir = viewdir[:, None].repeat(1, self.cascade_samples[0], 1)
        raw = self.forward_network(xyz, xyz_dir, ts_coarse, self.nerf)
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
            ts_fine = ts[:, None].repeat(1, self.cascade_samples[0] + self.cascade_samples[1])
            raw = self.forward_network(xyz, xyz_dir, ts_fine, self.nerf_fine)
            ret = utils.raw2outputs(raw, z_vals, rays_d, cfg.task_arg.white_bkgd)

            for key in ret:
                output[key + '_1'] = ret[key]
            return output
        else:
            return ret


    def batchify_rays(self, rays, ts, batch):
        all_ret = {}
        chunk = cfg.task_arg.chunk_size
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], ts[i: i+chunk], batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret

    def forward(self, batch):    
        # B, N_rays, C = batch['rays'].shape
        # ret = self.batchify_rays(batch['rays'].reshape(-1, C), batch['ts'].reshape(-1) ,batch)
        # return {k:ret[k].reshape(B, N_rays, -1) for k in ret}
        # try:
        if len(batch['rays'].shape) == 3:
            B, N_rays, C = batch['rays'].shape
            ret = self.batchify_rays(batch['rays'].reshape(-1, C), batch['ts'].reshape(-1) ,batch)
            return {k:ret[k].reshape(B, N_rays, -1) for k in ret}
        else:
            N_rays, C = batch['rays'].shape
            ret = self.batchify_rays(batch['rays'], batch['ts'] ,batch)
            return ret
        # except:
        #     import ipdb; ipdb.set_trace()
        # return {k:ret[k].reshape(N_rays, -1) for k in ret}
        # return ret
        # 接下来到 /mnt/data/cxy_colmap/LearningNeRF_nerfw/lib/train/losses/nerfw.py/NetworkWrapper.forward
