import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from model.utils.block import ResBlock1, ResBlock2
from model.utils.common import get_padding
from typing import List, Optional

LRELU_SLOPE = 0.1

class Generator(nn.Module):
    def __init__(self, hidden_channels: int, upsample_initial_channel: int, upsample_rates: List[int], upsample_kernel_sizes: List[int], resblock_kernel_sizes: List[int], resblock_dilation_sizes: List[List[int]], resblock: int = 1, gin_channels: Optional[int] = None) -> None:
        super().__init__()
        assert resblock == 1 or resblock == 2
        if resblock == 1:
            resblock = ResBlock1
        else:
            resblock = ResBlock2

        self.n_layers = len(upsample_rates)
        self.n_blocks = len(resblock_kernel_sizes)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        self.pre_conv = nn.Conv1d(in_channels=hidden_channels, out_channels=upsample_initial_channel, kernel_size=7, padding=3)

        for i in range(self.n_layers):
            in_channels = upsample_initial_channel // (2**i)
            out_channels = upsample_initial_channel // (2**(i+1))
            k = upsample_kernel_sizes[i]
            u = upsample_rates[i]
            self.ups.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(
                        in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=u, padding=(k-u)//2
                    )
                )
            )
            for j in range(self.n_blocks):
                self.resblocks.append(
                    resblock(
                        out_channels, resblock_kernel_sizes[j], resblock_dilation_sizes[j]
                    )
                )
        
        self.post_conv = nn.Conv1d(in_channels=upsample_initial_channel//(2**self.n_layers), out_channels=1, kernel_size=7, padding=3)

        if gin_channels is not None:
            self.cond_layer = nn.Conv1d(in_channels=gin_channels, out_channels=upsample_initial_channel, kernel_size=1)
    
    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
        x = self.pre_conv(x)
        if g is not None:
            g = self.cond_layer(g)
            x = x + g
        for i in range(self.n_layers):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None

            for j in range(self.n_blocks):
                if xs is None:
                    xs = self.resblocks[i*self.n_blocks + j](x)
                else:
                    xs = xs + self.resblocks[i*self.n_blocks + j](x)
            x = xs / self.n_blocks
        
        x = F.leaky_relu(x)
        x = self.post_conv(x)
        x = torch.tanh(x)
        return x
    
    def remove_weight_norm(self):
        for layer in self.ups:
            remove_parametrizations(layer, 'weight')
        for layer in self.resblocks:
            layer.remove_weight_norm()
    
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = nn.utils.parametrizations.weight_norm if use_spectral_norm == False else nn.utils.parametrizations.spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = nn.utils.parametrizations.weight_norm if use_spectral_norm == False else nn.utils.parametrizations.spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs