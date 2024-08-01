import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from scipy.io import wavfile
import librosa
import json
from typing import List

MAX_AUDIO_VALUE = 32768

class YourTTSTargetProcessor:
    def __init__(self,
                 speaker_path: str, 
                 sampling_rate: int = 22050, n_mels: int = 80, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 256, fmin: float = 0, fmax: float = 8000, device: str = 'cpu') -> None:
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
        ).to(device)

        self.resampler = Resample(orig_freq=sampling_rate, new_freq=16000).to(device)

        self.hann_window = torch.hann_window(window_length=win_length).to(device)

        self.num_pad = (self.n_fft - self.hop_length) // 2

        with open(speaker_path, 'r') as file:
            self.speaker_dict = json.load(file)

        self.device = device

    def load_audio(self, path: str) -> torch.Tensor:
        sr, signal = wavfile.read(path)
        signal = signal / MAX_AUDIO_VALUE
        if sr != self.sampling_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sampling_rate)
        return torch.tensor(signal, dtype=torch.float, device=self.device)
    
    def mel_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.device != self.device:
            signal = signal.to(self.device)
        signal = F.pad(signal.unsqueeze(1), (self.num_pad, self.num_pad), mode='reflect')
        signal = signal.squeeze(1)

        spec = torch.view_as_real(torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.hann_window,
                      center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True))

        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

        spec = torch.matmul(self.mel_basis, spec)
        spec = torch.log(torch.clamp_min(spec, min=1e-6))

        return spec
    
    def resample_audio(self, signal: torch.Tensor) -> torch.Tensor:
        return self.resampler(signal)
    
    def __call__(self, signals: List[torch.Tensor], speakers: List[str]):
        lengths = []
        max_length = 0

        for signal in signals:
            length = len(signal)
            if max_length < length:
                max_length = length
            lengths.append(length)

        padded_signals = []
        speaker_ids = []
        for index, signal in enumerate(signals):
            padded_signals.append(
                F.pad(signal, (0, max_length - lengths[index]), value=0.0)
            )
            speaker_ids.append(self.speaker_dict[speakers[index]])

        padded_signals = torch.stack(padded_signals)
        lengths = torch.tensor(lengths)

        mels = self.mel_spectrogram(padded_signals)
        lengths = (lengths // self.hop_length)
        speaker_ids = torch.tensor(speaker_ids)

        return padded_signals, mels, lengths, speaker_ids
