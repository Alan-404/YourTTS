import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math

from pytorch_metric_learning.losses import ArcFaceLoss

class YourTTSCriterion:
    def __init__(self, n_speakers: int, dim: int) -> None:
        self.arc_face = ArcFaceLoss(n_speakers, dim)
        self.l_1 = nn.L1Loss()

    def reconstruction_loss(self, mel: torch.Tensor, mel_hat: torch.Tensor):
        loss = self.l_1(mel_hat, mel) * 45
        return loss
        
    def duration_loss(self, l_length: torch.Tensor):
        return torch.sum(l_length.float())

    def feature_loss(self, fmap_r: List[torch.Tensor], fmap_g: List[torch.Tensor]):
        loss = 0.0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))
        return loss
        
    def generator_loss(self, disc_outputs: List[torch.Tensor]):
        loss = 0
        for dg in disc_outputs:
            dg = dg.float()
            l = torch.mean((1-dg)**2)
            loss += l
        return loss

    def kl_loss(self, z_p: torch.Tensor, m_p: torch.Tensor, logs_p: torch.Tensor, logs_q: torch.Tensor, mask: Optional[torch.Tensor] = None):
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        mask = mask.float()

        kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
        if mask is not None:
            kl = kl * mask
        kl = torch.sum(kl)
        
        if mask is not None:
            kl = kl / torch.sum(mask)

        return kl
        
    def discriminator_loss(self, disc_real_output: List[torch.Tensor], disc_generated_output: List[torch.Tensor]):
        loss = 0
        for dr, dg in zip(disc_real_output, disc_generated_output):
            dr = dr.float()
            dg = dg.float()

            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)

            loss += (r_loss + g_loss)
        
        return loss
    
    def speaker_consistency_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return 1 - F.cosine_similarity(x1, x2)
    
    def softmax_margin_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        return self.arc_face(outputs, labels)
    
class YourTTSMetric:
    def __init__(self) -> None:
        self.alpha = (10 * math.sqrt(2)) / (math.log(10))
    
    def mel_cepstral_distortion(self, output: torch.Tensor, label: torch.Tensor) -> float:
        '''
            output, label: (batch_size, n_mel_channels, mel_frame)
        '''
        distance = torch.mean(torch.sqrt(torch.sum(torch.pow((label - output), 2), dim=1)), dim=1)
        return torch.mean(self.alpha * distance).item()
    
    def signal_noise_ratio(self, y: torch.Tensor, y_hat: torch.Tensor):
        p_signal = torch.mean(torch.pow(y, 2), dim=1)
        p_noise = torch.mean(torch.pow((y - y_hat), 2), dim=1)
        return torch.mean(10 * torch.log10(p_signal / p_noise))