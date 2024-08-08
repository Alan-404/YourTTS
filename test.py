import torch

from model.your_tts import YourTTS
from manager import CheckpointManager
from processing.processor import YourTTSProcessor
from typing import List, Optional

def test(
        test_path: str,
        checkpoint: str,
        rank: int,
        world_size: int,
        # dataset config
        # checkpoint config
        save_checkpoint_after_epochs: int = 3,
        n_saved_checkpoints: int = 3,
        num_train_samples: Optional[int] = None,
        # tokenizer config
        tokenizer_path: str = "./tokenizer/vietnamese.json",
        pad_token: str = "<PAD>", 
        delim_token: str = "|", 
        unk_token: str = "<UNK>", 
        sampling_rate: int = 22050, 
        num_mels: int = 80, 
        n_fft: int = 1024, 
        hop_length: int = 256, 
        win_length: int = 1024, 
        fmin: float = 0.0, 
        fmax: float = 8000.0,
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
    