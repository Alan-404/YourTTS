import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.encoder import TextEncoder, PosteriorEncoder
from model.modules.duration import StochasticDurationPredictor
from model.modules.speaker import SpeakerEncoder
from model.modules.flow import Flow
from model.modules.vocoder import Generator
from model.utils.masking import generate_mask
from model.utils.common import rand_slice_segments
from model.modules.mas.search import find_path
import numpy as np
import math

from typing import Optional, List

class YourTTS(nn.Module):
    def __init__(self,
                 token_size: int,
                 n_mel_channels: int,
                 d_model: int = 192,
                 n_blocks: int = 10,
                 n_heads: int = 2,
                 kernel_size: int = 3,
                 hidden_channels: int = 192,
                 upsample_initial_channel: int = 512,
                 upsample_rates: List[int] = [8,8,2,2],
                 upsample_kernel_sizes: List[int] = [16,16,4,4],
                 resblock_kernel_sizes: List[int] = [3,7,11],
                 resblock_dilation_sizes: List[List[int]] = [[1,3,5], [1,3,5], [1,3,5]],
                 gin_channels: Optional[int] = 256,
                 dropout_p: float = 0.0,
                 segment_size: Optional[int] = 8192) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_channels = hidden_channels
        if segment_size is not None:
            self.segment_size = segment_size // np.prod(upsample_rates)
        else:
            self.segment_size = None

        self.text_encoder = TextEncoder(
            token_size=token_size,
            n_blocks=n_blocks,
            d_model=d_model,
            n_heads=n_heads,
            kernel_size=kernel_size,
            dropout_p=dropout_p
        )

        self.projection = nn.Conv1d(in_channels=hidden_channels, out_channels=2*hidden_channels, kernel_size=1)

        self.posterior_encoder = PosteriorEncoder(
            n_mel_channels=n_mel_channels,
            hidden_channels=d_model,
            out_channels=hidden_channels,
            kernel_size=5,
            n_layers=16,
            dilation_rate=1,
            gin_channels=gin_channels
        )
        
        self.flow = Flow(
            n_flows=4, in_channels=hidden_channels, hidden_channels=hidden_channels, kernel_size=5, dilation_rate=1, n_layers=4, gin_channels=gin_channels
        )

        self.duration_predictor = StochasticDurationPredictor(
            in_channels=d_model,
            hidden_channels=hidden_channels,
            kernel_size=3,
            n_flows=4,
            dropout_p=0.1,
            gin_channels=gin_channels
        )

        self.decoder = Generator(
            hidden_channels=hidden_channels,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            gin_channels=gin_channels
        )

        self.speaker_encoder = SpeakerEncoder(proj_dim=gin_channels)

    def forward(self, x: torch.Tensor, g: torch.Tensor, y: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, y_lengths: Optional[torch.Tensor] = None):        
        g = self.speaker_encoder(g)

        x_mask = None
        if x_lengths is not None:
            x_mask = generate_mask(x_lengths).unsqueeze(dim=1)
        y_mask = None
        if y_lengths is not None:
            y_mask = generate_mask(y_lengths).unsqueeze(dim=1)

        h_text = self.text_encoder(x, x_mask if x_mask is not None else None)
        text_stats = self.projection(h_text)
        if x_mask is not None:
            text_stats = text_stats * x_mask
        m_p, logs_p = torch.split(text_stats, [self.hidden_channels]*2, dim=1)
 
        z, m_q, logs_q = self.posterior_encoder(y)
        if y_mask is not None:
            z = z * y_mask
        
        z_p, _ = self.flow(z, mask=y_mask, g=g)

        with torch.no_grad():
            s_pq = torch.exp(-2*logs_p) # (batch_size, hidden_channels, text_length)

            neg_cent_1 = torch.sum(-0.5*math.log(2*math.pi) - logs_p, dim=1, keepdim=True) # (batch_size, 1, text_length)
            neg_cent_2 = torch.matmul((-0.5 * (z_p**2)).transpose(-1, -2), s_pq) # (batch_size, mel_length, text_length)
            neg_cent_3 = torch.matmul(z_p.transpose(-1, -2), (m_p * s_pq)) # (batch_size, mel_length, text_length)
            neg_cent_4 = torch.sum(-0.5 * (m_p**2) * s_pq, dim=1, keepdim=True) # (batch_size, 1, text_length)

            neg_cent = neg_cent_1 + neg_cent_2 + neg_cent_3 + neg_cent_4
            attn = find_path(neg_cent, text_lengths=x_lengths, mel_lengths=y_lengths).unsqueeze(1).detach()

        l_length = self.duration_predictor(h_text, w=attn.sum(dim=2), mask=x_mask, g=g)
        if x_mask is not None:
            l_length = l_length / torch.sum(x_mask)
        else:
            l_length = l_length / (x.size(0) * x.size(-1))
        
        attn = attn.squeeze(1)
        m_p = torch.matmul(attn, m_p.transpose(-1, -2)).transpose(1,2)
        logs_p = torch.matmul(attn, logs_p.transpose(-1, -2)).transpose(1,2)

        if self.segment_size is not None:
            sliced_z, sliced_indexes = rand_slice_segments(z, y_lengths, self.segment_size)
            o = self.decoder(sliced_z, g=g)
        else:
            o = self.decoder(z, g=g)
            sliced_indexes = None

        return o, l_length, sliced_indexes, x_mask, y_mask, z, z_p, m_p, logs_p, m_q, logs_q, g
    
    def infer(self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, sid: Optional[torch.Tensor] = None, length_scale: int = 1, noise_scale: float = 1.0, max_len: Optional[int] = None):
        batch_size = x.size(0)
        
        if x_lengths is not None:
            x_mask = generate_mask(x_lengths).unsqueeze(dim=1)
        else:
            x_mask = torch.ones((x.size(0), 1, x.size(1)), dtype=bool, device=x.device)
        
        if sid is not None:
            g = self.speaker_encoder(sid).unsqueeze(-1)
        else:
            g = None
        
        h_text = self.text_encoder(x, x_mask if x_mask is not None else None)
        stats = self.projection(h_text)
        if x_mask is not None:
            stats = stats * x_mask
        
        m_p, logs_p = torch.split(stats, [self.hidden_channels]*2, dim=1)
        logw = self.duration_predictor(h_text, mask=x_mask, g=g, reverse=True, noise_scale=noise_scale)

        w = torch.exp(logw)
        if x_mask is not None:
            w = w * x_mask
        w = w * length_scale

        w_ceil = torch.ceil(w)
        y_length = torch.clamp_min(torch.sum(w_ceil, dim=[1,2]), min=1)
        y_mask = generate_mask(y_length).unsqueeze(1)

        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)

        path = self.generate_path(w_ceil, attn_mask).squeeze(1)

        m_p = torch.matmul(path, m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(path, logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn((batch_size, self.hidden_channels, m_p.size(-1)), dtype=m_p.dtype, device=m_p.device) * torch.exp(logs_p)
        z_p = z_p * y_mask

        z = self.flow(z_p, mask=y_mask, reverse=True, g=g) * y_mask

        o = self.decoder(z, g=g)

        return o

    def generate_path(self, duration: torch.Tensor, mask: torch.Tensor):
        batch_size, _, t_y, t_x = mask.size()

        cum_duration = torch.cumsum(duration, dim=-1)
        cum_duration_flat = cum_duration.view(batch_size * t_x)

        path = generate_mask(cum_duration_flat, t_y).to(dtype=duration.dtype)
        path = path.view(batch_size, t_x, t_y)
        path = path - F.pad(path, self.convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
        path = path.unsqueeze(1).transpose(2, 3) * mask
        return path

    def convert_pad_shape(self, pad_shape):
        l = pad_shape[::-1]
        pad_shape = [item for sublist in l for item in sublist]
        return pad_shape

    def remove_weight_norm(self):
        self.flow.remove_weight_norm()
        self.decoder.remove_weight_norm()
