import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
import math

from typing import Optional, Callable, Union

class LinearSpectrogram(nn.Module):
    def __init__(self,
                 n_fft: int,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window_fn: Callable[..., torch.Tensor] = torch.hann_window,
                 pad: int = 0,
                 center: bool = False,
                 normalized: Union[bool, str] = False,
                 pad_mode: str = 'reflect',
                 onesided: bool = True) -> None:
        super().__init__()
        self.num_pad = (n_fft - hop_length) // 2

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (self.num_pad, self.num_pad), mode='reflect')

class MelScale(nn.Module):
    def __init__(self,
                 n_stft: int,
                 n_mels: int = 128,
                 sample_rate: int = 16000,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 norm: Optional[str] = None,
                 mel_scale: str = 'htk') -> None:
        super().__init__()
        assert mel_scale in ['htk', 'slaney']

        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2

        self.norm = norm
        self.n_mels = n_mels
        self.n_stft = n_stft
        self.mel_scale = mel_scale

        if self.mel_scale == 'slaney':
            self.f_sp = 200.0 / 3
            self.logstep = 27.0 / math.log(6.4)

        self.register_buffer('filterbank', self.mel_filterbank())

    def hz_to_mel(self, freq: float) -> float:
        if self.mel_scale == 'htk':
            return 2595.0 * math.log10(1.0 + (freq / 700.0))
        else:
            if freq < 1000:
                return freq / self.f_sp
            else:
                return 15.0 + (math.log(freq / 1000) * self.logstep)
            
    def mel_to_hz(self, mels: torch.Tensor) -> torch.Tensor:
        if self.mel_scale == 'htk':
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        else:
            freqs = mels * self.f_sp
            is_over = (mels >= 15)
            freqs[is_over] = 1000.0 * torch.exp((mels[is_over] - 15.0) / self.logstep)
            return freqs
        
    def create_triangular_filterbank(self, all_freqs: torch.Tensor, f_pts: torch.Tensor):
        f_diff = f_pts[1:] - f_pts[:-1] # (n_mels - 1)
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1) # (n_stft, n_mels)

        zero = torch.zeros(1)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
        up_slopes = slopes[:, 2:] / f_diff[1:]
        fb = torch.max(zero, torch.min(down_slopes, up_slopes))

        return fb
    
    def mel_filterbank(self):
        all_freqs = torch.linspace(0, self.sample_rate // 2, self.n_stft)

        mel_min = self.hz_to_mel(self.f_min)
        mel_max = self.hz_to_mel(self.f_max)

        mel_pts = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        f_pts = self.mel_to_hz(mel_pts)

        filterbank = self.create_triangular_filterbank(all_freqs, f_pts)

        if self.norm is not None and self.norm == 'slaney':
            enorm = 2.0 / (f_pts[2 : self.n_mels + 2] - f_pts[: self.n_mels])
            filterbank = filterbank * enorm.unsqueeze(0)

        return filterbank # (n_stft, n_mels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            x: Spectrogram input, shape = [batch_size, n_stft, time]
            -----
            output: Mel - Spectrogram, shape = [batch_size, n_mels, time]
        '''
        x = torch.matmul(x.transpose(1, 2), self.filterbank).transpose(1, 2)
        return x
