import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import Optional, Callable, Union

class MelSpectrogram(nn.Module):
    def __init__(self,
                 sample_rate: int,
                 n_fft: int,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 n_mels: int = 128,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 window_fn: Callable[..., torch.Tensor] = torch.hann_window,
                 pad: int = 0,
                 center: bool = True,
                 power: Optional[float] = 2.0,
                 normalized: Union[bool, str] = False,
                 pad_mode: str = 'reflect',
                 onesided: bool = True,
                 norm: Optional[str] = None,
                 mel_scale: str = 'htk',
                 use_norm: bool = True) -> None:
        super().__init__()
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2

        self.f_max = f_max if f_max is not None else float(sample_rate // 2)

        self.spectrogram = Spectrogram(
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=window_fn,
            pad=pad,
            center=center,
            power=power,
            normalized=normalized,
            pad_mode=pad_mode,
            onesided=onesided
        )

        self.mel_scale = MelScale(
            n_stft= n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=self.f_max,
            norm=norm,
            mel_scale=mel_scale
        )

        self.use_norm = use_norm
        if self.use_norm:
            self.register_buffer("norm", torch.randn((n_mels)))

    def log_mel(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log(torch.clamp(x, min=1e-5))
        return x

    def forward(self, x: torch.Tensor, log_mel: bool = True) -> torch.Tensor:
        x = self.spectrogram(x)
        x = self.mel_scale(x)
        if log_mel:
            x = self.log_mel(x)
        if self.use_norm:
            x = x / self.norm.unsqueeze(0).unsqueeze(-1)
        return x

class Spectrogram(nn.Module):
    def __init__(self,
                 n_fft: int,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window_fn: Callable[..., torch.Tensor] = torch.hann_window,
                 pad: int = 0,
                 center: bool = True,
                 power: Optional[float] = 2.0,
                 normalized: Union[bool, str] = False,
                 pad_mode: str = 'reflect',
                 onesided: bool = True) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2

        self.pad = pad
        self.center = center
        self.power = power
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        self.register_buffer("window", window_fn(self.win_length))

        self.num_pad = int((self.n_fft - self.hop_length) / 2)
        self.frame_length_norm = False
        self.window_norm = False

        self.get_spec_norm()
    
    def get_spec_norm(self):
        if torch.jit.isinstance(self.normalized, str):
            if self.normalized == 'frame_length':
                self.frame_length_norm = True
            elif self.normalized == 'window':
                self.window_norm = True
        elif torch.jit.isinstance(self.normalized, bool):
            if self.normalized:
                self.window_norm = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            x: Waveform input, shape = [batch_size, time]
            ----
            output: Spectrogram shape = [batch_size, n_fft // 2 + 1, time]
        '''
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad), mode='constant')

        x = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.frame_length_norm,
            onesided=self.onesided,
            return_complex=True
        )

        if self.window_norm:
            x = x / self.window.power(2.0).sum().sqrt()
        
        if self.power is not None:
            x = x.abs().pow(self.power)

        return x

class MelScale(nn.Module):
    def __init__(self,
                 n_stft: int,
                 n_mels: int = 128,
                 sample_rate: int = 16000,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 norm: Optional[str] = None,
                 mel_scale: str = "htk") -> None:
        super().__init__()
        assert mel_scale in ['htk', 'slaney'], "Invalid Format of Scaling"

        self.sample_rate = sample_rate
        self.n_stft = n_stft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.norm = norm
        self.mel_scale = mel_scale
        self.f_sp = 200.0 / 3
        self.logstep = 27.0 / math.log(6.4)

        self.register_buffer('filterbank', self.mel_filterbank())

    def hz_to_mel(self, freq: float) -> float:
        if self.mel_scale == 'htk':
            return 2595.0 * math.log10(1.0 + (freq / 700))
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

class Resample(nn.Module):
    def __init__(self,
                 original_freq: int,
                 target_freq: int,
                 resampling_method: str = 'sinc_interp_hann',
                 lowpass_filter_width: int = 6,
                 rolloff: float = 0.99,
                 beta: Optional[float] = None,
                 dtype: Optional[torch.dtype] = torch.float,
                 device: str = 'cpu') -> None:
        super().__init__()
        self.original_freq = original_freq
        self.target_freq = target_freq
        self.resampling_method = resampling_method
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.beta = beta if beta is not None else 14.769656459379492

        self.dtype = dtype
        self.device = device

        self.gcd = math.gcd(int(original_freq), int(target_freq))

        if self.original_freq != self.target_freq:
            kernel, self.width = self.get_sinc_resample_kernel()
            self.register_buffer("kernel", kernel)

    def get_sinc_resample_kernel(self):
        orig_freq = int(self.original_freq) // self.gcd
        targ_freq = int(self.target_freq) // self.gcd

        base_freq = min(orig_freq, targ_freq)
        base_freq = base_freq * self.rolloff

        width = math.ceil(self.lowpass_filter_width * orig_freq / base_freq)
        
        idx = torch.arange(-width, width + orig_freq, dtype=torch.float, device=self.device)[None, None] / orig_freq

        t = torch.arange(0, -targ_freq, -1, dtype=self.dtype, device=self.device)[:, None, None] / targ_freq + idx
        t *= base_freq
        t = t.clamp_(-self.lowpass_filter_width, self.lowpass_filter_width)

        if self.resampling_method == "sinc_interp_hann":
            window = torch.cos(t * math.pi / self.lowpass_filter_width / 2) ** 2
        else:
            # sinc_interp_kaiser
            beta_tensor = torch.tensor(float(self.beta))
            window = torch.i0(beta_tensor * torch.sqrt(1 - (t / self.lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)

        t *= math.pi

        scale = base_freq / orig_freq
        kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
        kernels *= window * scale

        if self.dtype is None:
            kernels = kernels.to(dtype=torch.float32)

        return kernels, width
    
    def _apply_sinc_resample_kernel(self, waveform: torch.Tensor):
        if not waveform.is_floating_point():
            raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")

        orig_freq = int(self.original_freq) // self.gcd
        new_freq = int(self.target_freq) // self.gcd

        # pack batch
        shape = waveform.size()
        waveform = waveform.view(-1, shape[-1])

        num_wavs, length = waveform.shape
        waveform = torch.nn.functional.pad(waveform, (self.width, self.width + orig_freq))
        resampled = torch.nn.functional.conv1d(waveform[:, None], self.kernel, stride=orig_freq)
        resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
        target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
        resampled = resampled[..., :target_length]

        # unpack batch
        resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
        return resampled
    
    def forward(self, x: torch.Tensor):
        return self._apply_sinc_resample_kernel(x)


class PreEmphasis(nn.Module):
    def __init__(self, coefficient: float = 0.97) -> None:
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer('filter', torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x.unsqueeze(1), (1, 0), mode='reflect')
        x = F.conv1d(x, self.filter).squeeze(1)
        return x