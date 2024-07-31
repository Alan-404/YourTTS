import torch
from torch.utils.data import Dataset, Sampler, BatchSampler
import pandas as pd
from processing.processor import VITSProcessor
from tqdm import tqdm
from typing import Any, Optional, Tuple
import itertools
import random

class VITSDataset(Dataset):
    def __init__(self, manifest_path: str, processor: VITSProcessor, make_phoneme: bool = False, training: bool = True, num_examples: Optional[int] = None, is_multi_speakers: bool = False) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path)
        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]
        if 'duration' in self.prompts.columns:
            self.prompts = self.prompts.sort_values(by=['duration'], ignore_index=True)
        self.processor = processor

        self.columns = self.prompts.columns
        self.training = training

        if make_phoneme or "phoneme" not in self.columns:
            phonemes = []
            texts = self.prompts['text'].to_list()
            for item in tqdm(texts):
                phonemes.append(" ".join(self.processor.sentence2phonemes(item)))
            self.prompts['phoneme'] = phonemes

            self.prompts.to_csv(manifest_path, index=False)
        
        self.is_multi_speakers = is_multi_speakers
    
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index: int):
        index_df = self.prompts.iloc[index]

        phonemes = index_df['phoneme'].split(" ")

        if self.training:
            path = index_df['path']
            if self.is_multi_speakers:
                sid = index['user']
                return phonemes, self.processor.load_audio(path), sid

            return phonemes, self.processor.load_audio(path)
        else:
            return phonemes

class VITSSampler(Sampler):
    def __init__(self, data_source: VITSDataset) -> None:
        # super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        return iter(indices)
    
    def __len__(self):
        return len(self.data_source)
    
class VITSBatchSampler(BatchSampler):
    def __init__(self, sampler: VITSSampler, batch_size: int = 1, drop_last: bool = False) -> None:
        # super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield random.shuffle(batch)
                batch = []
        if len(batch) > 0 and self.drop_last == False:
            yield random.shuffle(batch)

    def __len__(self) -> int:
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class VITSCollate:
    def __init__(self, processor: VITSProcessor, training: bool = True) -> None:
        self.processor = processor
        self.training = training
    def __call__(self, batch: Tuple[torch.Tensor]) -> Any:
        if self.training:
            phonemes, signals = zip(*batch)
            tokens, token_lengths = self.processor(phonemes)
            signals, signal_lengths = self.processor.as_target(signals)

            return tokens, signals, token_lengths, signal_lengths
        else:
            tokens, lengths = self.processor(batch)
            return tokens, lengths