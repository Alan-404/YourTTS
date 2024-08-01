import os
import shutil

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import wandb
from typing import Dict, Optional, Tuple, Union, Literal

class EarlyStoppingManager:
    def __init__(self, n_patiences: int = 3, condition: Literal['higher', 'lower'] = 'lower') -> None:
        self.count = 0
        self.history = None

        self.n_patiences = n_patiences
        self.condition = True if condition == 'higher' else False
        self.tend = None

    def __call__(self, criterion: Union[torch.Tensor, float]) -> None:
        if self.history is None:
            self.history = criterion
            return

class CheckpointManager:
    def __init__(self, saved_folder: Optional[str] = None, n_savings: int = 3) -> None:
        self.saved_folder = saved_folder
        if os.path.exists(self.saved_folder) == False:
            os.makedirs(self.saved_folder)

        self.n_savings = n_savings

        self.saved_checkpoints = []

    def load_checkpoint(self, checkpoint_path: str, model: Module, optimizer: Optimizer, scheduler: LRScheduler) -> Tuple[int, int]:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        n_steps = checkpoint['n_steps']
        n_epochs = checkpoint['n_epochs']

        return n_steps, n_epochs
    
    def save_checkpoint(self, model: Module, optimizer: Optimizer, scheduler: LRScheduler, n_steps: int, n_epochs: int, filename: str = 'model') -> None:
        data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'n_steps': n_steps,
            'n_epochs': n_epochs
        }

        if len(self.saved_checkpoints) == self.n_savings:
            shutil.rmtree(f"{self.saved_folder}/{self.saved_checkpoints[0]}")
            self.saved_checkpoints.pop(0)

        checkpoint_folder = f"{self.saved_folder}/{n_steps}"
        if os.path.exists(checkpoint_folder) == False:
            os.makedirs(checkpoint_folder)

        torch.save(data, f"{checkpoint_folder}/{filename}.pt")
        self.saved_checkpoints.append(n_steps)

class LoggerManager:
    def __init__(self, project: str = "YourTTS", name: Optional[str] = None) -> None:
        self.project = project
        self.name = name

        wandb.init(project=project, name=name)

    def log_data(self, data: Dict, step: int) -> None:
        wandb.log(data, step)

    def log_image(self):
        pass