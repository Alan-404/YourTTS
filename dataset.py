import torch
from torch.utils.data import Dataset
import pandas as pd
from processing.processor import YourTTSProcessor
from processing.target import YourTTSTargetProcessor
from typing import Any, Optional, Tuple, Union

class YourTTSDataset(Dataset):
    def __init__(self, manifest_path: str, processor: YourTTSProcessor, handler: YourTTSTargetProcessor, training: bool = False, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path)
        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        # Remove speakers which have only one file audio
        value_counts = self.prompts['channel'].value_counts()
        mask = self.prompts['channel'].isin(value_counts[value_counts > 1].index)
        self.prompts = self.prompts[mask].reset_index(drop=True)

        self.processor = processor
        self.handler = handler

        self.training = training

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        index_df = self.prompts.iloc[index]
        path = index_df['path']
        speaker = index_df['channel']
        phonemes = index_df['phoneme'].split(" ")

        ref_path = self.prompts[(self.prompts['channel'] == speaker) & (self.prompts['path'] != path)].sample(1)['path'].to_list()[0]
        ref_audio = self.processor.load_audio(ref_path)

        tokens = self.processor.phonemes2tokens(phonemes)

        if self.training == False:
            return tokens, ref_audio

        signal = self.handler.load_audio(path)

        return tokens, ref_audio, signal, speaker
    
class YourTTSCollate:
    def __init__(self, processor: YourTTSProcessor, handler: YourTTSTargetProcessor, training: bool = False) -> None:
        self.processor = processor
        self.handler = handler

        self.training = training

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