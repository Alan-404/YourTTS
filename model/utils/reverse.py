import torch
import torch.nn as nn
from typing import Optional
from model.utils.block import WN

class ResidualCouplingLayer(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, n_layers: int, dilation_rate: int, dropout_p: float = 0.0, mean_only: bool = False, gin_channels: Optional[int] = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.half_channels = in_channels // 2
        self.mean_only = mean_only

        self.pre_conv = nn.Conv1d(in_channels=self.half_channels, out_channels=hidden_channels, kernel_size=1)
        self.wn = WN(channels=hidden_channels, n_layers=n_layers, dilation_rate=dilation_rate, kernel_size=kernel_size, dropout_p=dropout_p, gin_channels=gin_channels)
        self.post_conv = nn.Conv1d(in_channels=hidden_channels, out_channels=self.half_channels*(2 - mean_only), kernel_size=1)

        self.post_conv.weight.data.zero_()
        self.post_conv.bias.data.zero_()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, reverse: bool = False, g: Optional[torch.Tensor] = None):
        batch_size, _, time = x.size()
        
        x0, x1 = torch.split(x, [self.half_channels]*2, dim=1)
        h = self.pre_conv(x0)
        if mask is not None:
            h = h * mask
        h = self.wn(h, mask=mask, g=g)
        stats = self.post_conv(h)
        if mask is not None:
            stats = stats * mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels]*2, dim=1)
        else:
            m = stats
            logs = torch.zeros((batch_size, self.half_channels, time), dtype=x.dtype, device=x.device)
        
        if not reverse:
            x1 = x1 * torch.exp(logs) + m
            if mask is not None:
                x1 = x1 * mask
            x = torch.cat([x0, x1], dim=1)
            logdet = torch.sum(logs, dim=[1,2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs)
            if mask is not None:
                x1 = x1 * mask
            x = torch.cat([x0, x1], dim=1)
            return x
        
    def remove_weight_norm(self):
        self.wn.remove_weight_norm()

class ElementwiseAffine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.rand((channels, 1)))
        self.transition = nn.Parameter(torch.rand((channels, 1)))
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, reverse: bool = False):
        if not reverse:
            x = x*torch.exp(self.scale) + self.transition
            if mask is not None:
                x = x * mask
                logdet = self.scale * mask
            else:
                logdet = self.scale.unsqueeze(0)
            return x, torch.sum(logdet, dim=[1, 2])
        else:
            x = (x - self.transition) * torch.exp(-self.scale)
            return x

class Log(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, reverse: bool = False): 
        if not reverse:
            y = torch.log(torch.clamp_min(x, min=1e-5))
            if mask is not None:
                y = y * mask
            logdet = torch.sum(-y, dim=[1,2])
            return y, logdet
        else:
            x = torch.exp(x)
            if mask is not None:
                x = x * mask
            return x
        
class Flip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: torch.Tensor):
        x = torch.flip(x, dims=[1])
        return x