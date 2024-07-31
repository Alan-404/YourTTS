import torch
import torch.nn as nn
from model.utils.reverse import ResidualCouplingLayer, Flip

from typing import Optional

class Flow(nn.Module):
    def __init__(self, n_flows: int, in_channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, dropout_p: float = 0.0, gin_channels: Optional[int] = None) -> None:
        super().__init__()
        self.n_flows = n_flows

        self.residual_layers = nn.ModuleList()
        self.flips = nn.ModuleList()

        for _ in range(n_flows):
            self.residual_layers.append(
                ResidualCouplingLayer(in_channels=in_channels, hidden_channels=hidden_channels, kernel_size=kernel_size, n_layers=n_layers, dilation_rate=dilation_rate, dropout_p=dropout_p, mean_only=True, gin_channels=gin_channels)
            )
            self.flips.append(Flip())
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, reverse: bool = False, g: Optional[torch.Tensor] = None):
        if not reverse:
            logdet_total = 0
            for i in range(self.n_flows):
                x, logdet = self.residual_layers[i](x, mask=mask, reverse=reverse, g=g)
                logdet_total = logdet_total + logdet
                x = self.flips[i](x)
            return x, logdet_total
        else:
            for i in range(self.n_flows - 1, -1, -1):
                x = self.flips[i](x)
                x = self.residual_layers[i](x, mask=mask, reverse=reverse, g=g)
            return x
        
    def remove_weight_norm(self):
        for layer in self.residual_layers:
            layer.remove_weight_norm()