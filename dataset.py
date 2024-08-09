import torch
from torch.utils.data import Dataset
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from processing.processor import YourTTSProcessor
from processing.target import YourTTSTargetProcessor
import random
from typing import Optional, Tuple, Union, Dict

class YourTTSDataset(Dataset):
    def __init__(self, dataset: Union[pa.Table, str], processor: YourTTSProcessor, handler: Optional[YourTTSTargetProcessor] = None, training: bool = False, num_examples: Optional[int] = None) -> None:
        super().__init__()
        if isinstance(dataset, str):
            self.dataset = pq.read_table(dataset)
        else:
            self.dataset = dataset

        if num_examples is not None:
            self.dataset = self.dataset.slice(0, num_examples)

        self.processor = processor
        self.handler = handler

        self.training = training

        if self.training:
            assert self.handler is not None

    def __len__(self) -> int:
        return len(self.dataset)
    
    def query_by_speaker(self, speaker: str) -> pa.Table:
        condition = pc.equal(self.dataset, speaker)
        return self.dataset.filter(condition)
    
    def get_random_audio_sample(self, speaker: int, path: str) -> str:
        speaker_table = self.query_by_speaker(speaker)
        path_condition = pc.equal(speaker_table['path'], path)
        speaker_table = speaker_table.filter(pc.invert(path_condition))

        random_index = random.randint(0, speaker_table.num_rows)

        return self.get_row_by_index(random_index)['path']
    
    def get_row_by_index(self, index: int) -> Dict:
        return {col: self.dataset[col][index].as_py() for col in self.dataset.column_names}

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        row = self.get_row_by_index(index)

        speaker = row['speaker']
        text = row['text']

        if self.training:
            path = row['path']

            ref_path = self.get_random_audio_sample(speaker, path)

            audio = self.handler.load_audio(path)
            ref_audio = self.processor.load_audio(ref_path)

            return ref_audio, text, audio
        else:
            path = row['ref_path']
            ref_audio = self.processor.load_audio(path)

            return ref_audio, text
    
class YourTTSCollate:
    def __init__(self, processor: YourTTSProcessor, handler: Optional[YourTTSTargetProcessor] = None, training: bool = False) -> None:
        self.processor = processor
        self.handler = handler

        self.training = training

        if self.training:
            assert self.handler is not None

    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.training:
            tokens, ref_audios, signals, speakers = zip(*batch)

            tokens, ref_audios, token_lengths, _ = self.processor(tokens, ref_audios)
            signals, mels, mel_lengths, speakers = self.handler(signals, speakers)

            return tokens, ref_audios, mels, token_lengths, mel_lengths, signals, speakers
        else:
            tokens, ref_audios = zip(*batch)
            tokens, ref_audios, token_lengths, _ = self.processor(tokens, ref_audios)

            return tokens, ref_audios, token_lengths