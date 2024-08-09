import os

import torch
import torch.distributed as distributed

from model.your_tts import YourTTS
from manager import CheckpointManager
from processing.processor import YourTTSProcessor

import pandas as pd

from typing import List, Optional

def setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] =  '12355'
    distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    print(f"Initialized Thread at {rank + 1}/{world_size}")

def cleanup() -> None:
    distributed.destroy_process_group()

def test(
        rank: int,
        world_size: int,
        test_path: str,
        checkpoint: str,
        # dataset config
        # checkpoint config
        # tokenizer config
        tokenizer_path: str = "./tokenizer/vietnamese.json",
        pad_token: str = "<PAD>", 
        delim_token: str = "|", 
        unk_token: str = "<UNK>", 
        num_mels: int = 80, 
        # model config
        n_blocks: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        kernel_size: int = 3,
        hidden_channels: int = 192,
        upsample_initial_channel: int = 512,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        gin_channels: int = 512,
        dropout_p: float = 0.0
    ):

    if world_size > 1:
        setup(rank, world_size)
    
    checkpoint_manager = CheckpointManager()

    processor = YourTTSProcessor(
        tokenizer_path=tokenizer_path,
        pad_token=pad_token,
        delim_token=delim_token,
        unk_token=unk_token,
        sampling_rate=16000
    )

    model = YourTTS(
        token_size=len(processor.dictionary),
        n_mel_channels=num_mels,
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=n_heads,
        kernel_size=kernel_size,
        hidden_channels=hidden_channels,
        upsample_initial_channel=upsample_initial_channel,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        gin_channels=gin_channels,
        dropout_p=dropout_p
    ).to(rank)

    checkpoint_manager.load_checkpoint(checkpoint, model, only_weights=True)
    
    df = pd.read_csv(test_path)

    if world_size > 1:
        cleanup()