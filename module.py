import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Optional, List

from model.vits import VITS
from model.modules.encoder import PosteriorEncoder
from model.modules.audio import LinearSpectrogram
from model.utils.masking import generate_mask
from model.modules.mas.search import find_path
from processing.processor import VITSProcessor

class VITSModule(nn.Module):
    def __init__(self,
                 token_size: int,
                 sample_rate: int = 22050,
                 n_fft: int = 1024,
                 win_length: int = 1024,
                 hop_length: int = 256,
                 fmin: float = 0.0,
                 fmax: float = 8000.0,
                 n_mel_channels: int = 80,
                 d_model: int = 192,
                 n_blocks: int = 6,
                 n_heads: int = 2,
                 kernel_size: int = 3,
                 hidden_channels: int = 192,
                 upsample_initial_channel: int = 512,
                 upsample_rates: List[int] = [8,8,2,2],
                 upsample_kernel_sizes: List[int] = [16,16,4,4],
                 resblock_kernel_sizes: List[int] = [3,7,11],
                 resblock_dilation_sizes: List[List[int]] = [[1,3,5], [1,3,5], [1,3,5]],
                 dropout_p: float = 0.1,
                 segment_size: Optional[int] = 8192,
                 n_speakers: Optional[int] = None,
                 gin_channels: Optional[int] = None) -> None:
        super().__init__()

        self.hop_length = hop_length

        self.linear_spectrogram = LinearSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            fmin=fmin,
            fmax=fmax
        )

        self.posterior_encoder = PosteriorEncoder(
            n_mel_channels=n_mel_channels,
            hidden_channels=d_model,
            out_channels=hidden_channels,
            kernel_size=5,
            n_layers=16,
            dilation_rate=1,
            gin_channels=gin_channels
        )

        self.vits = VITS(
            token_size=token_size,
            d_model=d_model,
            n_blocks=n_blocks,
            n_heads=n_heads,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            dropout_p=dropout_p,
            segment_size=segment_size,
            n_speakers=n_speakers,
            gin_channels=gin_channels
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, y_lengths: Optional[torch.Tensor] = None, sid: Optional[torch.Tensor] = None):
        if sid is not None:
            g = self.vits.speaker_embedding(sid).unsqueeze(-1)
        else:
            g = None
        
        x_mask = None
        if x_lengths is not None:
            x_mask = generate_mask(x_lengths)

        with torch.no_grad():
            with autocast(enabled=False):
                linear_spec = self.linear_spectrogram(y.float())
        y_mask = None
        if y_lengths is not None:
            y_mask = generate_mask(y_lengths // self.hop_length)

        z, m_q, logs_q = self.posterior_encoder(linear_spec, y_mask, g)

        h_text = self.text_encoder(x, x_mask if x_mask is not None else None)
        text_stats = self.projection(h_text)