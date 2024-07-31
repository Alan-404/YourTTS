import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.encoder import ConditionEncoder
from model.utils.convolution import ConvFlow
from model.utils.reverse import Log, Flip, ElementwiseAffine
import math
from typing import Optional

class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, n_flows: int = 4, dropout_p: float = 0.0, gin_channels: Optional[int] = None) -> None:
        super().__init__()
        self.condition_encoder_h = ConditionEncoder(in_channels=in_channels, hidden_channels=hidden_channels, kernel_size=kernel_size, dropout_p=dropout_p)
        self.condition_encoder_d = ConditionEncoder(in_channels=1, hidden_channels=hidden_channels, kernel_size=kernel_size, dropout_p=dropout_p)

        self.log_flow = Log()

        self.posterior_encoder = DurationFlow(n_layers=n_flows, in_channels=2, hidden_channels=hidden_channels, kernel_size=kernel_size)
        self.flow = DurationFlow(n_layers=n_flows, in_channels=2, hidden_channels=hidden_channels, kernel_size=kernel_size)
        
        if gin_channels is not None:
            self.cond_layer = nn.Conv1d(in_channels=gin_channels, out_channels=hidden_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, reverse: bool = False, noise_scale: float = 1.0, g: Optional[torch.Tensor] = None):
        batch_size, _, time = x.size()
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            g = self.cond_layer(g)
        x = self.condition_encoder_h(x, mask=mask, g=g)

        if not reverse:
            assert w is not None
            logdet_total_q = 0
            h_w = self.condition_encoder_d(w, mask=mask, g=None)

            e_q = torch.randn((batch_size, 2, time), dtype=x.dtype, device=x.device)
            if mask is not None:
                e_q = e_q * mask
            z_q = e_q
            z_q, logdet_posterior = self.posterior_encoder(z_q, mask, reverse=reverse, g=x+h_w)
            logdet_total_q = logdet_total_q + logdet_posterior

            z_u, z1 = torch.split(z_q, [1,1], dim=1)
            u = F.sigmoid(z_u)
            if mask is not None:
                u = u * mask
            z0 = (w - u)
            if mask is not None:
                z0 = z0 * mask
            
            logdet_sigmoid = z_u - 2 * torch.log(1 + torch.exp(z_u))
            if mask is not None:
                logdet_sigmoid = logdet_sigmoid * mask
            logdet_total_q = logdet_total_q + torch.sum(logdet_sigmoid, dim=[1,2])

            logdet_e_q = -0.5 * (math.log(2*math.pi) + (e_q**2))
            if mask is not None:
                logdet_e_q = logdet_e_q * mask
            logdet_q = torch.sum(logdet_e_q, dim=[1,2]) - logdet_total_q

            logdet_total = 0
            z0, logdet_log = self.log_flow(z0, mask=mask)
            logdet_total = logdet_total + logdet_log
            z = torch.cat([z0, z1], dim=1)
            z, logdet_flow = self.flow(z, mask=mask, g=x, reverse=reverse)
            logdet_total = logdet_total + logdet_flow

            logdet_z = 0.5 * (math.log(2*math.pi) + (z**2))
            if mask is not None:
                logdet_z = logdet_z * mask

            nll = torch.sum(logdet_z, dim=[1,2]) - logdet_total

            return nll + logdet_q
        else:
            z = torch.randn((batch_size, 2, time), dtype=x.dtype, device=x.device) * noise_scale
            z = self.flow(z, reverse=reverse, g=g, mask=mask)
            z0, z1 = torch.split(z, [1,1], dim=1)
            return z0


class DurationFlow(nn.Module):
    def __init__(self, n_layers: int, in_channels: int, hidden_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.n_layers = n_layers

        self.affine = ElementwiseAffine(channels=in_channels)
        self.conv_flows = nn.ModuleList()
        self.flips = nn.ModuleList()

        for _ in range(n_layers):
            self.conv_flows.append(
                ConvFlow(in_channels=in_channels, hidden_channels=hidden_channels, kernel_size=kernel_size, n_layers=3)
            )
            self.flips.append(Flip())
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, reverse: bool = False, g: Optional[torch.Tensor] = None):
        if not reverse:
            logdet_total = 0.0
            x, logdet_affine = self.affine(x, mask=mask, reverse=reverse)
            logdet_total = logdet_total + logdet_affine

            for i in range(self.n_layers):
                x, logdet_conv = self.conv_flows[i](x, mask=mask, reverse=reverse, g=g)
                logdet_total = logdet_total + logdet_conv
                x = self.flips[i](x)
            return x, logdet_total
        else:
            for i in range(self.n_layers-1, -1, -1):
                x = self.flips[i](x)   
                x = self.conv_flows[i](x, mask=mask, reverse=reverse, g=g)
            x = self.affine(x, mask=mask, reverse=reverse)
            return x