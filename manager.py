import os
import shutil

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import wandb
import matplotlib.pyplot as plt
from typing import Dict, Optional


class CheckpointManager:
    def __init__(self, saved_folder: Optional[str] = None, n_saved: int = 3) -> None:
        self.saved_folder = saved_folder
        if os.path.exists(self.saved_folder):
            os.makedirs(self.saved_folder)

        self.n_saved = n_saved

        self.saved_checkpoints = []

    def load_checkpoint(self, checkpoint_path: str, model: Module, optimizer: Optimizer, scheduler: LRScheduler):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        n_steps = checkpoint['n_steps']
        n_epochs = checkpoint['n_epochs']

        return n_steps, n_epochs
    
    def save_checkpoint(self, model: Module, optimizer: Optimizer, scheduler: LRScheduler, n_steps: int, n_epochs: int, filename: str = 'model'):
        data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'n_steps': n_steps,
            'n_epochs': n_epochs
        }

        if len(self.saved_checkpoints) == self.n_saved:
            shutil.rmtree(f"{self.saved_folder}/{self.saved_checkpoints[0]}")
            self.saved_checkpoints.pop(0)

        checkpoint_folder = f"{self.saved_folder}/{n_steps}"
        if os.path.exists(checkpoint_folder) == False:
            os.makedirs(checkpoint_folder)

        torch.save(data, f"{checkpoint_folder}/{filename}.pt")
        self.saved_checkpoints.append(n_steps)

class LoggerManager:
    def __init__(self, project: str = "VITS", name: str = "TTS") -> None:
        self.project = project
        self.name = name

        wandb.init(project=project, name=name)

    def log_data(self, data: Dict, step: int):
        wandb.log(data, step)

    def log_image(self):
        pass