import torch
from torch.utils.data import Dataset

import pandas as pd
from processing.processor import VITSProcessor
from glob import glob
import random
from mapping import map_weights
from model.vits import VITS

from typing import Optional

class CloningInferenceDataset(Dataset):
    def __init__(self, path: str, processor: VITSProcessor, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.prompts = pd.read_csv(path)
        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]
        
        self.processor = processor

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index: int):
        index_df = self.prompts.iloc[index]

        text = index_df['text']
        folder_path = index_df['ref']
        audio_path = random.choice(glob(f"{folder_path}/*.wav"))

        phonemes = self.processor.sentence2phonemes(text)
        signal = self.processor.load_audio(audio_path)

        return phonemes, signal
    
def infer(path: str, 
          checkpoint: str,
          tokenizer_path: str,
          pad_token: str = "<PAD>",
          delim_token: str = "|",
          unk_token: str = "<UNK>",
          sampling_rate: int = 22050,
          num_mels: int = 80,
          n_fft: int = 1024,
          win_length: int = 1024,
          hop_length: int = 256,
          fmin: float = 0.0,
          fmax: float = 8000.0,):
    processor = VITSProcessor(
        sampling_rate=sampling_rate,
        num_mels=num_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        path=tokenizer_path,
        pad_token=pad_token,
        delim_token=delim_token,
        unk_token=unk_token
    )
