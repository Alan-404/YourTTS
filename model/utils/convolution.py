import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.utils.common import get_padding
from model.utils.block import LayerNormGELU
from model.utils.transform import piecewise_rational_quadratic_transform

from typing import Optional

class ConvFlow(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, n_layers: int, n_bins: int = 10, tail_bound: float = 5.0) -> None:
        super().__init__()
        self.half_channels = in_channels // 2
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        self.sqrt_dim = math.sqrt(hidden_channels)
        
        self.pre_conv = nn.Conv1d(in_channels=self.half_channels, out_channels=hidden_channels, kernel_size=1)
        self.dds_conv = DDSConv(n_layers=n_layers, channels=hidden_channels, kernel_size=kernel_size, dropout_p=0.0)
        self.post_conv = nn.Conv1d(in_channels=hidden_channels, out_channels=self.half_channels*(3 * n_bins - 1), kernel_size=1)

        self.post_conv.weight.data.zero_()
        self.post_conv.bias.data.zero_()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, reverse: bool = False, g: Optional[torch.Tensor] = None):
        batch_size, _, time = x.size()
        
        x0, x1 = torch.split(x, [self.half_channels]*2, dim=1)
        h = self.pre_conv(x0)
        if mask is not None:
            h = h * mask
        h = self.dds_conv(h, mask=mask, g=g)
        h = self.post_conv(h)
        if mask is not None:
            h = h * mask
        
        h = h.reshape((batch_size, self.half_channels, 3*self.n_bins - 1, time)).permute((0, 1, 3, 2))
        
        unnormalized_widths = h[..., :self.n_bins] / self.sqrt_dim
        unnormalized_heights = h[..., self.n_bins:2*self.n_bins] / self.sqrt_dim
        unnormalized_derivatives = h[..., 2*self.n_bins:]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            inputs=x1,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=reverse,
            tails='linear',
            tail_bound=self.tail_bound
        )

        x = torch.cat([x0, x1], dim=1)
        if mask is not None:
            x = x * mask
            logabsdet = logabsdet * mask
        if not reverse:
            return x, torch.sum(logabsdet, dim=[1,2])
        else:
            return x

class DDSConv(nn.Module):
    def __init__(self, n_layers: int, channels: int, kernel_size: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                DDSConvLayer(channels=channels, kernel_size=kernel_size, dilation=kernel_size**i, dropout_p=dropout_p)
            )
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, g: Optional[torch.Tensor] = None):
        if g is not None:
            x = x + g
        if mask is not None:
            x = x * mask
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DDSConvLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dilated_group_conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, dilation=dilation, padding=get_padding(kernel_size, dilation), groups=channels)
        self.norm_1 = LayerNormGELU(dim=channels)
        self.sep_conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.norm_2 = LayerNormGELU(dim=channels)
        self.dropout_p = dropout_p
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        xt = self.dilated_group_conv(x)
        xt = self.norm_1(xt)
        xt = self.sep_conv(xt)
        xt = self.norm_2(xt)
        xt = F.dropout(xt, p=self.dropout_p, training=self.training)
        x = x + xt
        if mask is not None:
            x = x * mask
        return x