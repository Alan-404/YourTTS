import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.utils.block import WN, EncoderBlock
from model.utils.convolution import DDSConv
import math
class TextEncoder(nn.Module):
    def __init__(self, token_size: int, n_blocks: int, d_model: int, n_heads: int, kernel_size: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks

        self.embedding = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)
        self.encoder = EncoderBlock(
            n_layers=n_blocks,
            hidden_channels=d_model,
            n_heads=n_heads,
            kernel_size=kernel_size,
            dropout_p=dropout_p
        )

        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**(-0.5))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(-1, -2)
        if mask is not None:
            x = x * mask
        x = self.encoder(x, mask)
        return x
    
class PosteriorEncoder(nn.Module):
    def __init__(self, n_mel_channels: int, hidden_channels: int, out_channels: int, n_layers: int, kernel_size: int, dilation_rate: int, dropout_p: float = 0.0, gin_channels: Optional[int] = None) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.pre_conv = nn.Conv1d(in_channels=n_mel_channels, out_channels=hidden_channels, kernel_size=1)
        self.wn = WN(channels=hidden_channels, n_layers=n_layers, dilation_rate=dilation_rate, kernel_size=kernel_size, dropout_p=dropout_p, gin_channels=gin_channels)
        self.proj = nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels * 2, kernel_size=1)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, length = x.size()
        x = self.pre_conv(x)
        if mask is not None:
            x = x * mask
        x = self.wn(x, mask=mask, g=g)
        if mask is not None:
            x = x * mask
        stats = self.proj(x)
        m, logs = torch.split(stats, [self.out_channels] * 2, dim=1)
        z = m + torch.randn((batch_size, self.out_channels, length), dtype=x.dtype, device=x.device) * torch.exp(logs) # Reparameterization Trick
        return z, m, logs
    
class ConditionEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.pre_conv = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.dds_conv = DDSConv(n_layers=3, channels=hidden_channels, kernel_size=kernel_size, dropout_p=dropout_p)
        self.post_conv = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, g: Optional[torch.Tensor] = None):
        x = self.pre_conv(x)
        if g is not None:
            x = x + g
        x = self.dds_conv(x, mask=mask, g=None)
        x = self.post_conv(x)
        if mask is not None:
            x = x * mask
        return x