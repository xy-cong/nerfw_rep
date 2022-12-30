import torch
import torch.nn.functional as F
from lib.config import cfg
TINY_NUMBER = 1e-6



def aabb_intersection(rays, wbounds):
    bounds = wbounds.reshape((2, 3)).to(rays.device)
    ray_o, ray_d = rays[..., :3], rays[..., 3:6]

    nominator = bounds[None] - ray_o[:, None]
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]

    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = torch.norm(ray_d, dim=1)
    d0 = torch.norm(p_intervals[:, 0] - ray_o, dim=1) / norm_ray
    d1 = torch.norm(p_intervals[:, 1] - ray_o, dim=1) / norm_ray
    near = torch.minimum(d0, d1)
    far = torch.maximum(d0, d1)

    return near, far, mask_at_box

def sample_along_ray(near, far, N_samples):
    z_steps = torch.linspace(0, 1, N_samples, device=near.device)[None]
    z_vals = near[..., None] * (1 - z_steps) + far[..., None] * z_steps
    # xyz = rays_o[:, :, None] + rays_d[:, :, None] * z_vals[:, :, :, None]
    return z_vals

def raw2outputs(raw, z_vals, rays_d, white_bkgd=False):
    # import ipdb; ipdb.set_trace() 
    raw2alpha = lambda raw, dists, act_fn=F.softplus: 1.-torch.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(raw.device)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])
    alpha = raw2alpha(raw[...,3], dists)

    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[..., :-1]
    T = torch.cat([torch.ones_like(T[..., 0:1]), T], dim=-1)
    weights = alpha * T

    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)

    if white_bkgd:
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[...,None])
    ret = {'rgb': rgb_map, 'depth': depth_map, 'weights': weights}
    return ret

def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    return z_vals

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # B = len(bins)
    # bins, weights = bins.reshape(-1, bins.shape[-1]), weights.reshape(-1, weights.shape[-1])
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

# def sample_pdf(bins, weights, N_samples, det=False):
#     '''
#     :param bins: tensor of shape [..., M+1], M is the number of bins
#     :param weights: tensor of shape [..., M]
#     :param N_samples: number of samples along each ray
#     :param det: if True, will perform deterministic sampling
#     :return: [..., N_samples]
#     '''
#     # Get pdf
#     weights = weights + TINY_NUMBER      # prevent nans
#     pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [..., M]
#     cdf = torch.cumsum(pdf, dim=-1)                             # [..., M]
#     cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)     # [..., M+1]

#     # Take uniform samples
#     dots_sh = list(weights.shape[:-1])
#     M = weights.shape[-1]

#     min_cdf = 0.00
#     max_cdf = 1.00       # prevent outlier samples

#     if det:
#         u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
#         u = u.view([1]*len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples,])   # [..., N_samples]
#     else:
#         sh = dots_sh + [N_samples]
#         u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf        # [..., N_samples]

#     # Invert CDF
#     # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
#     above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

#     # random sample inside each bin
#     below_inds = torch.clamp(above_inds-1, min=0)
#     inds_g = torch.stack((below_inds, above_inds), dim=-1)     # [..., N_samples, 2]

#     cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])   # [..., N_samples, M+1]
#     cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)       # [..., N_samples, 2]

#     bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])    # [..., N_samples, M+1]
#     bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

#     # fix numeric issue
#     denom = cdf_g[..., 1] - cdf_g[..., 0]      # [..., N_samples]
#     denom = torch.where(denom<TINY_NUMBER, torch.ones_like(denom), denom)
#     t = (u - cdf_g[..., 0]) / denom

#     samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

#     return samples
