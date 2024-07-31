import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from processing.processor import VITSProcessor
from model.vits import VITS
from typing import List, Optional
from scipy.io import wavfile
import torchsummary
import fire
from tqdm import tqdm
from dataset import VITSDataset, VITSCollate
import pandas as pd

from processing.processor import VITSProcessor

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print(f"Setup Thread {rank + 1}/{world_size}")

def cleanup():
    dist.destroy_process_group()

def infer(
        # Thread Config
        rank: int,
        world_size: int,
        data_path: str,
        saved_folder: str,
        # Checkpoint Config
        checkpoint: str,
        # Processor Config
        tokenizer_path: str,
        pad_token: str = "<PAD>",
        delim_token: str = "|",
        unk_token: str = "<UNK>",
        sampling_rate: int = 22050,
        num_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        fmin: float = 0.,
        fmax: float = 8000.,
        # Model Config
        n_blocks: int = 6,
        d_model: int = 192,
        n_heads: int = 2,
        kernel_size: int = 3,
        hidden_channels: int = 192,
        upsample_initial_channel: int = 512,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        # Dataset Config
        num_samples: Optional[int] = None,
        batch_size: int = 1
    ):
    assert os.path.exists(checkpoint) and os.path.isfile(checkpoint), "Checkpoint is unvalid"
    if rank == 0:
        if os.path.exists(saved_folder) == False:
            os.makedirs(f"{saved_folder}/audios")
    if world_size > 1:
        setup(rank, world_size)

    processor = VITSProcessor(
        path=tokenizer_path,
        pad_token=pad_token,
        delim_token=delim_token,
        unk_token=unk_token,
        sampling_rate=sampling_rate,
        num_mels=num_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        device=rank
    )

    model = VITS(
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
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes
    )

    if rank == 0:
        torchsummary.summary(model)

    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
    model.to(rank)
    model.remove_weight_norm()
    model.eval()

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    dataset = VITSDataset(manifest_path=data_path, processor=processor, training=False, num_examples=num_samples)
    collate_fn = VITSCollate(processor=processor, training=False)
    data_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, collate_fn=collate_fn)

    audio_paths = []

    for index, (x, lengths) in enumerate(tqdm(dataloader, leave=False)):
        with torch.no_grad():
            o = model.infer(x, lengths)
            filename = f"{saved_folder}/audios/{index}.wav"
            wavfile.write(filename, rate=processor.sampling_rate, data=o[0].numpy())
            audio_paths.append(filename)

    if world_size > 1:
        gathered_audio_paths = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_audio_paths, audio_paths)
        audio_paths = [item for sublist in gathered_audio_paths for item in sublist]

    if rank == 0:
        df = pd.DataFrame({
            'text': dataset.prompts['text'].to_list(),
            'path': audio_paths
        })
        df.to_csv(f"{saved_folder}/result.csv")

    if world_size > 1:
        cleanup()

    print("Finish The Inference")

def main(
        data_path: str,
        saved_folder: str,
        # Checkpoint Config
        checkpoint: str,
        # Processor Config
        tokenizer_path: str,
        pad_token: str = "<PAD>",
        delim_token: str = "|",
        unk_token: str = "<UNK>",
        sampling_rate: int = 22050,
        num_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        fmin: float = 0.,
        fmax: float = 8000.,
        # Model Config
        n_blocks: int = 6,
        d_model: int = 192,
        n_heads: int = 2,
        kernel_size: int = 3,
        hidden_channels: int = 192,
        upsample_initial_channel: int = 512,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        # Dataset Config
        num_samples: Optional[int] = None,
        batch_size: int = 1
    ):
    n_gpus = torch.cuda.device_count()
    assert n_gpus == 0

    if n_gpus == 1:
        infer(
            rank=0,
            world_size=n_gpus,
            data_path=data_path,
            saved_folder=saved_folder,
            checkpoint=checkpoint,
            tokenizer_path=tokenizer_path,
            pad_token=pad_token,
            delim_token=delim_token,
            unk_token=unk_token,
            sampling_rate=sampling_rate,
            num_mels=num_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            n_blocks=n_blocks,
            d_model=d_model,
            n_heads=n_heads,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            num_samples=num_samples,
            batch_size=batch_size
        )
    else:
        mp.spawn(
            infer,
            args=(
                n_gpus,
                data_path,
                saved_folder,
                checkpoint,
                tokenizer_path,
                pad_token,
                delim_token,
                unk_token,
                sampling_rate,
                num_mels,
                n_fft,
                hop_length,
                win_length,
                fmin,
                fmax,
                n_blocks,
                d_model,
                n_heads,
                kernel_size,
                
            )
        )

if __name__ == '__main__':
    fire.Fire(main)