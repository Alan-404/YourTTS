import os

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from processing.processor import VITSProcessor

from dataset import VITSDataset, VITSCollate
from model.vits import VITS

from typing import Tuple, List, Optional
from scipy.io import wavfile

from mapping import map_weights

import fire

def infer(path: str,
          checkpoint: str,
          tokenizer_path: str,
          exported_folder: str = "./results",
          batch_size: int = 1,
          # Processor Config
          pad_token: str = "<PAD>",
          delim_token: str = "|",
          unk_token: str = "<UNK>",
          sampling_rate: int = 22050,
          num_mels: int = 80,
          n_fft: int = 1024,
          win_length: int = 1024,
          hop_length: int = 256,
          fmin: float = 0.0,
          fmax: float = 8000.0,
          # Model Config
          d_model: int = 192,
          n_heads: int = 2,
          n_blocks: int = 6,
          hidden_channels: int = 192,
          upsample_initial_channel: int = 512,
          upsample_rates: List[int] = [8, 8, 2, 2],
          upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
          resblock_kernel_sizes: List[int] = [3, 7, 11],
          resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
          n_speakers: Optional[int] = None,
          gin_channels: Optional[int] = None,
          segment_size: int = 8192,
          device: str = 'cuda'):

    assert os.path.exists(f"{exported_folder}/audios") == False, "Existed Audios in Folder"
    
    audio_folder = f"{exported_folder}/audios"
    
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
        unk_token=unk_token,
        device=device
    )

    model = VITS(
        token_size=len(processor.dictionary),
        n_mel_channels=processor.num_mels,
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=n_heads,
        hidden_channels=hidden_channels,
        upsample_initial_channel=upsample_initial_channel,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        dropout_p=0.,
        n_speakers=n_speakers,
        gin_channels=gin_channels,
        segment_size=segment_size
    )
    
    model.load_state_dict(map_weights(torch.load(checkpoint, map_location='cpu')['state_dict']))
    model.to(device)
    model.remove_weight_norm()
    model.eval()
    
    collate_fn = VITSCollate(processor=processor)
    dataset = VITSDataset(path, processor=processor, training=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    paths = []

    for index, data in enumerate(tqdm(dataloader)):
        tokens, lengths = data

        tokens = tokens.to(device)
        lengths = lengths.to(device)

        with torch.inference_mode():
            signals = model.infer(tokens, lengths).squeeze(1).numpy()
        
        count = 0
        for item in signals:
            audio_path = f"{audio_folder}/{index + count}.wav"
            wavfile.write(audio_path, rate=processor.sampling_rate, data=item)
            paths.append(audio_path)
            count += 1

    dataframe = dataset.prompts
    dataframe['path'] = paths

    dataframe.to_csv(f"{exported_folder}/output.csv", index=False)

if __name__ == '__main__':
    fire.Fire(infer)