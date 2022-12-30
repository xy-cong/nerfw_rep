import torch
import torch.nn as nn
import torch.nn.functional as F

def get_geo2alpha(mode):
    if mode == 'nerf':
        return nerf_geo2alpha
    elif 'volsdf' in mode:
        return volsdf_geo2alpha()
    else:
        raise NotImplementedError

def nerf_geo2alpha(geo, z_vals, rays_d):
    raw2alpha = lambda raw, dists, act_fn=F.softplus: 1.-torch.exp(-act_fn(raw-10.)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = dists * rays_d[:, 0].norm(dim=-1, keepdim=True)
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(geo.device)], -1)
    alpha = raw2alpha(geo[..., 0], dists)
    return alpha


class volsdf_geo2alpha(nn.Module):
    def __init__(self):
        super(volsdf_geo2alpha, self).__init__()
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, geo, z_vals, rays_d):
        x = -geo
        ind0 = x <= 0
        val0 = 1 / self.beta * (0.5 * torch.exp(x[ind0] / self.beta))

        ind1 = x > 0
        val1 = 1 / self.beta * (1 - 0.5 * torch.exp(-x[ind1] / self.beta))

        val = torch.zeros_like(geo)
        val[ind0] = val0
        val[ind1] = val1
        return nerf_geo2alpha(val, z_vals, rays_d)

