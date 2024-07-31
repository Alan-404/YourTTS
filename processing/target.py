import torch
import torch.nn.functional as F
from scipy.io import wavfile
import librosa

from typing import List

MAX_AUDIO_VALUE = 32768

class YourTTSTargetProcessor:
    def __init__(self, sampling_rate: int = 22050, n_mels: int = 80, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 256, fmin: float = 0, fmax: float = 8000, device: str = 'cpu') -> None:
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_mels = n_mels
        self.win_length = win_length

        self.mel_basis = torch.from_numpy(
            librosa.filters.mel(
                sr=sampling_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax
            )
        )

        self.hann_window = torch.hann_window(window_length=win_length).to(device)

        self.device = device

    def load_audio(self, path: str) -> torch.Tensor:
        sr, signal = wavfile.read(path)
        signal = signal / MAX_AUDIO_VALUE
        if sr != self.sampling_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sampling_rate)

        return torch.tensor(signal, dtype=torch.float)
    
    def mel_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.device != self.device:
            signal = signal.to(self.device)
        signal = F.pad(signal.unsqueeze(1), (int((self.n_fft-self.hop_length)/2), int((self.n_fft-self.hop_length)/2)), mode='reflect')
        signal = signal.squeeze(1)

        spec = torch.view_as_real(torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.hann_window,
                      center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True))

        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

        spec = torch.matmul(self.mel_basis, spec)
        spec = torch.log(torch.clamp_min(spec, min=1e-5))

        return spec
    
    def __call__(self, signals: List[torch.Tensor]):
        lengths = []
        max_length = 0

        for signal in signals:
            length = len(signal)
            if max_length < length:
                max_length = length
            lengths.append(length)

        padded_signals = []
        for index, signal in enumerate(signals):
            padded_signals.append(
                F.pad(signal, (0, max_length - lengths[index]), value=0.0)
            )

        padded_signals = torch.stack(padded_signals)
        lengths = torch.tensor(lengths)

        mels = self.mel_spectrogram(padded_signals)
        lengths = (length // self.hop_length) + 1

        return padded_signals, mels, lengths
