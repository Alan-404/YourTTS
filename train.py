import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DistributedSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb.plot

from manager import CheckpointManager
from dataset import VITSDataset, VITSCollate
from processing.processor import VITSProcessor
from model.vits import VITS
from model.modules.vocoder import MultiPeriodDiscriminator
from evaluation import VITSCriterion
from model.utils.common import slice_segments

from tqdm import tqdm
import statistics
from typing import Optional, List
import shutil
import wandb

import fire

# manual_seed = 1234

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print(f"Initialize Thread at {rank+1}/{world_size}")

def cleanup():
    dist.destroy_process_group()

def clip_grad_value_(parameters: torch.Tensor, clip_value: Optional[float] = None, norm_type: int = 2) -> float:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def train(rank: int,
          world_size: int,
          # dataset config
          train_path: str,
          checkpoint: Optional[str] = None,
          saved_checkpoints: str = "./checkpoints",
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
          d_model: int = 192,
          n_heads: int = 2,
          kernel_size: int = 3,
          hidden_channels: int = 192,
          upsample_initial_channel: int = 512,
          upsample_rates: List[int] = [8, 8, 2, 2],
          upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
          resblock_kernel_sizes: List[int] = [3, 7, 11],
          resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
          dropout_p: float = 0.1,
          segment_size: int = 8192,
          # train config
          batch_size: int = 1,
          num_epochs: int = 1,
          lr: float = 2e-4,
          set_lr: bool = False,
          fp16: bool = False,
          # Logger Config
          logging: bool = True,
          project_name: str = 'VITS',
          username: Optional[str] = None
    ):
    if world_size > 1:
        setup(rank, world_size)
    # torch.manual_seed(manual_seed)

    checkpoint_manager = CheckpointManager(saved_checkpoints, n_saved_checkpoints)

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

    if rank == 0:
        n_steps = 0
        current_epoch = 0
        if logging:
            wandb.init(
                project=project_name,
                name=username
            )

    mel_frame = segment_size // processor.hop_length

    generator = VITS(
        token_size=len(processor.dictionary),
        n_mel_channels=processor.num_mels,
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
        dropout_p=dropout_p
    ).to(rank)
    discriminator = MultiPeriodDiscriminator().to(rank)

    if world_size > 1:
        generator = DDP(generator, device_ids=[rank])
        discriminator = DDP(discriminator, device_ids=[rank])

    gen_optim = optim.AdamW(params=generator.parameters(), lr=lr, betas=[0.8, 0.99], weight_decay=0.01)
    disc_optim = optim.AdamW(params=discriminator.parameters(), lr=lr, betas=[0.8, 0.99], weight_decay=0.01)

    gen_scheduler = lr_scheduler.ExponentialLR(optimizer=gen_optim, gamma=0.999875)
    disc_scheduler = lr_scheduler.ExponentialLR(optimizer=disc_optim, gamma=0.999875)

    if checkpoint is not None and os.path.exists(checkpoint):
        loaded_steps, loaded_epoch = checkpoint_manager.load_checkpoint(f"{checkpoint}/vits.pt", generator, gen_optim, gen_scheduler)
        checkpoint_manager.load_checkpoint(f"{checkpoint}/disc.pt", discriminator, disc_optim, disc_scheduler)
        if rank == 0:
            current_epoch = loaded_epoch
            n_steps = loaded_steps
    
    if set_lr:
        gen_optim.param_groups[0]['lr'] = lr
        disc_optim.param_groups[0]['lr'] = lr

    collate_fn = VITSCollate(processor=processor, training=True)

    dataset = VITSDataset(train_path, processor=processor, training=True, num_examples=num_train_samples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    criterion = VITSCriterion()

    scaler = GradScaler(enabled=fp16)
    saved_index = save_checkpoint_after_epochs - 1

    if rank == 0:
        print("Start Traning")
        print("=====================================\n")

    for epoch in range(num_epochs):
        if world_size > 1:
            dataloader.sampler.set_epoch(epoch)
        if rank == 0:
            print(f"Epoch: {epoch}")
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_gen_loss = 0.0
        train_duration_loss = 0.0
        train_fm_loss = 0.0
        
        train_disc_loss = 0.0

        g_grad_norm = 0.0
        d_grad_norm = 0.0

        generator.train()
        discriminator.train()
        for _, (x, y, x_lengths, y_lengths) in enumerate(tqdm(dataloader, leave=False)):
            x = x.to(rank)
            y = y.to(rank)
            x_lengths = x_lengths.to(rank)
            y_lengths = y_lengths.to(rank)
            
            with autocast(enabled=fp16):
                y_hat, mels, l_length, sliced_indexes, _, mel_mask, _, z_p, m_p, logs_p, _, logs_q = generator(x, y, x_lengths, y_lengths)
                
                mel_hat = processor.mel_spectrogram(y_hat.squeeze(1))
                mel_truth = slice_segments(mels, sliced_indexes, mel_frame)
                y = slice_segments(y.unsqueeze(1), sliced_indexes * processor.hop_length, segment_size)

                y_dp_hat_r, y_dp_hat_g, _, _ = discriminator(y, y_hat.detach())
                
                with autocast(enabled=False):
                    disc_loss = criterion.discriminator_loss(y_dp_hat_r, y_dp_hat_g)
                    assert torch.isnan(disc_loss) == False
                
            disc_optim.zero_grad()
            scaler.scale(disc_loss).backward()
            scaler.unscale_(disc_optim)
            disc_grad_norm = clip_grad_value_(discriminator.parameters(), None)
            scaler.step(disc_optim)

            with autocast(enabled=fp16):
                _, y_dp_hat_g, fmap_dp_r, fmap_dp_g = discriminator(y, y_hat)
                with autocast(enabled=False):
                    recon_loss = criterion.reconstruction_loss(mel_truth, mel_hat)
                    kl_loss = criterion.kl_loss(z_p, m_p, logs_p, logs_q, mel_mask)
                    dur_loss = criterion.duration_loss(l_length)
                    gen_loss = criterion.generator_loss(y_dp_hat_g)
                    fm_loss = criterion.feature_loss(fmap_dp_r, fmap_dp_g)

                    vae_loss = recon_loss + kl_loss + dur_loss + gen_loss + fm_loss
                    assert torch.isnan(vae_loss) == False
            
            gen_optim.zero_grad()
            scaler.scale(vae_loss).backward()
            scaler.unscale_(gen_optim)
            generator_grad_norm = clip_grad_value_(generator.parameters(), None)
            scaler.step(gen_optim)

            scaler.update()

            train_recon_loss += recon_loss
            train_kl_loss += kl_loss
            train_gen_loss += gen_loss
            train_duration_loss += dur_loss
            train_fm_loss += fm_loss

            train_disc_loss += disc_loss

            g_grad_norm += generator_grad_norm
            d_grad_norm += disc_grad_norm

            if rank == 0:
                n_steps += 1
        
        # Schedulers step
        gen_scheduler.step()
        disc_scheduler.step()

        num_batches = len(dataloader)

        train_recon_loss = train_recon_loss / num_batches
        train_kl_loss = train_kl_loss / num_batches
        train_duration_loss = train_duration_loss / num_batches
        train_gen_loss = train_gen_loss / num_batches
        train_fm_loss = train_fm_loss / num_batches

        train_disc_loss = train_disc_loss / num_batches

        g_grad_norm = g_grad_norm / num_batches
        d_grad_norm = d_grad_norm / num_batches
        
        if rank == 0:
            elbo_loss = train_recon_loss + train_kl_loss
            vae_loss = elbo_loss + train_duration_loss + train_gen_loss + train_fm_loss
            current_lr = gen_optim.param_groups[0]['lr']
            print(f'Reconstruction Loss: {(train_recon_loss):.4f}')
            print(f"KL Loss: {(train_kl_loss):.4f}")
            print(f"Duration Loss: {(train_duration_loss):.4f}")
            print(f"Generation Loss: {(train_gen_loss):.4f}")
            print(f"Feature Map Loss: {(train_fm_loss):.4f}")
            print("-----------------------------------------")
            print(f"ELBO Loss: {(elbo_loss):.4f}")
            print(f"VAE Loss: {(vae_loss):.4f}")
            print("=========================================")
            print(f"Discriminator Loss: {(train_disc_loss):.4f}")
            print("=========================================")
            print(f"Current Learning Rate: {current_lr}")
            print("=========================================")
            print(f"Generator Gradient Norm: {(g_grad_norm):.4f}")
            print(f"Discriminator Gradient Norm: {(d_grad_norm):.4f}")
            print("\n")

            wandb.log({
                'recon_loss': train_recon_loss,
                'kl_loss': train_kl_loss,
                'duration_loss': train_duration_loss,
                'generation_loss': train_gen_loss,
                'feature_map_loss': train_fm_loss,
                'elbo_loss': elbo_loss,
                'vae_loss': vae_loss,
                'discriminator_loss': train_disc_loss,
                'learning_rate': current_lr,
                'generator_gradient_norm': g_grad_norm,
                'discriminator_gradient_norm': d_grad_norm
            }, n_steps)

            current_epoch += 1

            if epoch % save_checkpoint_after_epochs == saved_index or epoch == num_epochs - 1:
                checkpoint_manager.save_checkpoint(generator, gen_optim, gen_scheduler, n_steps, current_epoch, 'vits')
                checkpoint_manager.save_checkpoint(discriminator, disc_optim, disc_scheduler, n_steps, current_epoch, 'disc')
                print(f"Checkpoint is saved at {saved_checkpoints}/{n_steps}\n")
    
    if world_size > 1:
        cleanup()
            
def main(
        # dataset config
        train_path: str,
        checkpoint: Optional[str] = None,
        saved_checkpoints: str = "./checkpoints",
        save_checkpoint_after_epochs: int = 3,
        n_saved_checkpoints: int = 3,
        num_train_samples: Optional[int] = None,
        # processor config
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
        d_model: int = 192,
        n_heads: int = 2,
        kernel_size: int = 3,
        hidden_channels: int = 192,
        upsample_initial_channel: int = 512,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        dropout_p: float = 0.1,
        segment_size: int = 8192,
        # train config
        batch_size: int = 1,
        num_epochs: int = 1,
        lr: float = 2e-4,
        set_lr: bool = False,
        fp16: bool = False,
        # Logger Config
        logging: bool = True,
        project_name: str = 'VITS',
        username: Optional[str] = None
    ):
    if torch.cuda.is_available() == False:
        raise("CUDA is Not Available")
    n_gpus = torch.cuda.device_count()
    if os.path.exists(saved_checkpoints) == False:
        os.mkdir(saved_checkpoints)
    if n_gpus == 1:
        train(
            0, n_gpus, train_path,
            checkpoint, saved_checkpoints, save_checkpoint_after_epochs, n_saved_checkpoints,
            num_train_samples, tokenizer_path, pad_token, delim_token, unk_token,
            sampling_rate, num_mels, n_fft, hop_length, win_length, fmin, fmax,
            n_blocks, d_model, n_heads, kernel_size, hidden_channels, upsample_initial_channel, upsample_rates, upsample_kernel_sizes,
            resblock_kernel_sizes, resblock_dilation_sizes, dropout_p, segment_size,
            batch_size, num_epochs, lr, set_lr, bool(fp16),
            logging, project_name, username
        )
    else:
        mp.spawn(
            train,
            args=(
                n_gpus, train_path,
                checkpoint, saved_checkpoints, save_checkpoint_after_epochs, n_saved_checkpoints,
                num_train_samples, tokenizer_path, pad_token, delim_token, unk_token,
                sampling_rate, num_mels, n_fft, hop_length, win_length, fmin, fmax,
                n_blocks, d_model, n_heads, kernel_size, hidden_channels, upsample_initial_channel, upsample_rates, upsample_kernel_sizes,
                resblock_kernel_sizes, resblock_dilation_sizes, dropout_p, segment_size,
                batch_size, num_epochs, lr, set_lr, bool(fp16),
                logging, project_name, username
            ),
            nprocs=n_gpus,
            join=True
        )

if __name__ == '__main__':
    fire.Fire(main)
