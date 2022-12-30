import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg

class NeRF(nn.Module):
    def __init__(self, input_ch, input_ch_views, cfg):
        """
        """
        super(NeRF, self).__init__()
        D, W, V_D = cfg.D, cfg.W, cfg.V_D
        skips, use_viewdirs = [4], True
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W)] + [nn.Linear(W, W)  for i in range(V_D)])

        if use_viewdirs:
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            h = torch.cat([h, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
