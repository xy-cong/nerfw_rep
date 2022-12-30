import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg

class NeRF(nn.Module):
    def __init__(self, input_ch, input_ch_views, typ, cfg):
        """
        """
        super(NeRF, self).__init__()
        D, W, V_D = cfg.D, cfg.W, cfg.V_D
        skips, use_viewdirs = [4], True
        self.typ = typ
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # ------------------------------------------------- nerfa ------------------------------------------------- #
        self.appearance_encode = False if typ == 'coarse' else cfg.encode_a
        self.input_ch_appearance = cfg.N_a if cfg.encode_a else 0
        # ------------------------------------------------- nerfa ------------------------------------------------- #

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        # import ipdb; ipdb.set_trace()
        # self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W)] + [nn.Linear(W, W)  for i in range(V_D)])

        # if use_viewdirs:
        #     self.alpha_linear = nn.Linear(W, 1)
        #     self.rgb_linear = nn.Linear(W, 3)
        # else:
        #     self.output_linear = nn.Linear(W, output_ch)

        # ------------------------------------------------- nerfw ------------------------------------------------- #
        self.xyz_encoding_final = nn.Linear(W, W)
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W + self.input_ch_appearance, W//2)])

        if use_viewdirs:
            self.static_sigma = nn.ModuleList([nn.Linear(W, 1)])
            self.static_rgb = nn.ModuleList([nn.Linear(W//2, 3)])   
        # ------------------------------------------------- nerfw ------------------------------------------------- #



    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        input_pts, input_dir_a = torch.split(x, [self.input_ch, self.input_ch_views+self.input_ch_appearance], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # alpha = self.alpha_linear(h)
            # import ipdb; ipdb.set_trace()
            static_sigma = nn.Softplus(self.static_sigma[0](h)).beta
            
            xyz_encoding_final = self.xyz_encoding_final(h)
            dir_encoding = torch.cat([xyz_encoding_final, input_dir_a], 1)

            for i, l in enumerate(self.views_linears):
                dir_encoding = self.views_linears[i](dir_encoding)
                dir_encoding = F.relu(dir_encoding)
            # import ipdb; ipdb.set_trace()
            static_rgb = nn.Softmax(self.static_rgb[0](dir_encoding)).dim
            static = torch.cat([static_rgb,static_sigma], 1)
            outputs = static

        else:
            outputs = self.output_linear(h)

        return outputs
