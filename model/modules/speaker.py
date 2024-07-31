import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from model.modules.audio import MelSpectrogram, PreEmphasis
from typing import Optional, List

class SpeakerEncoder(nn.Module):
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_fft: int = 512,
                 win_length: Optional[int] = 400,
                 hop_length: Optional[int] = 160,
                 pre_emphasis_coeff: float = 0.97,
                 input_dim: int = 64, 
                 proj_dim: int = 512,
                 layers: List[int] = [3, 4, 6, 3],
                 num_filters: List[int]  = [32, 64, 128, 256],
                 encoder_type: str = 'ASP') -> None:
        super().__init__()
        assert encoder_type in ['ASP', 'SAP']
        self.encoder_type = encoder_type
        self.input_dim = input_dim

        self.pre_emphasis = PreEmphasis(coefficient=pre_emphasis_coeff)
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=input_dim,
            use_norm=False
        )

        self.instancenorm = nn.InstanceNorm1d(input_dim)
        
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.layers = nn.ModuleList()
        for i in range(len(num_filters)):
            if i == 0:
                stride = 1
                in_dim = num_filters[0]
            else:
                stride = (2, 2)
                in_dim = num_filters[i-1]
            self.layers.append(
                ResNetLayer(in_dim, num_filters[i], layers[i], stride=stride, expansion=1)
            )
        
        outmap_size = input_dim // 8

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[-1] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[-1] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2)
        )

        if self.encoder_type == 'SAP':
            out_dim = num_filters[-1] * outmap_size
        else:
            out_dim = num_filters[-1] * outmap_size * 2
        
        self.fc = nn.Linear(out_dim, proj_dim)
    
    def forward(self, x: torch.Tensor, l2_norm: bool = False) -> torch.Tensor:
        with torch.no_grad():
            with autocast(enabled=False):
                x = self.pre_emphasis(x)
                x = self.mel_spectrogram(x, log_mel=True)
        x = self.instancenorm(x).unsqueeze(1) # (batch_size, 1, input_dim, length)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size()[0], -1, x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)
        
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        if l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

class ResNetLayer(nn.Module):
    def __init__(self, inplanes: int, planes: int, n_blocks: int, stride: int = 1, expansion: int = 1) -> None:
        super().__init__()
        downsample = None
        if inplanes != planes * expansion and stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion)
            )
        
        self.layers = nn.ModuleList()
        self.layers.append(SpeakerEncoderBasisBlock(inplanes, planes, stride=stride, downsample=downsample))

        for _ in range(1, n_blocks):
            self.layers.append(
                SpeakerEncoderBasisBlock(planes * expansion, planes, stride=1)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class SpeakerEncoderLayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)
        y = self.fc(y).unsqueeze(-1).unsqueeze(-1)
        x = x * y
        return x
    
class SpeakerEncoderBasisBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, reduction: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SpeakerEncoderLayer(planes, reduction)
        self.downsample = downsample if downsample is not None else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        residual = self.downsample(x)
    
        out = out + residual
        out = self.relu(out)
        return out