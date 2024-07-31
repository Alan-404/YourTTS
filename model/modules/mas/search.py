import torch
from typing import Optional
import numpy as np
from model.modules.mas import core

def find_path(neg_cent: torch.Tensor, text_lengths: Optional[torch.Tensor] = None, mel_lengths: Optional[torch.Tensor] = None):
    batch_size, mel_len, text_len = neg_cent.size()
    dtype = neg_cent.dtype
    device = neg_cent.device

    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    paths = np.zeros_like(neg_cent, dtype=np.int32)
    
    if text_lengths is not None and mel_lengths is not None:
        text_lengths = text_lengths.data.cpu().numpy().astype(np.int32)
        mel_lengths = mel_lengths.data.cpu().numpy().astype(np.int32)
    else:
        text_lengths = np.array([text_len] * batch_size)
        mel_lengths = np.array([mel_len] * batch_size)

    core.maximum_path_c(paths, neg_cent, mel_lengths, text_lengths)

    return torch.from_numpy(paths).to(device=device, dtype=dtype)