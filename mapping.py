import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict

import os

def map_weights(checkpoint: OrderedDict, name: str = "model"):
    data = OrderedDict()
    for key in checkpoint.keys():
        if name in key:
            data[key] = checkpoint[key]
    data = OrderedDict((key.replace(f"{name}.", ""), value) for key, value in data.items())
    return data

def load_weights(model: torch.nn.Module, weights: OrderedDict):
    return model.load_state_dict(weights)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint_dict['model'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    scheduler.load_state_dict(checkpoint_dict['scheduler'])

    steps = checkpoint_dict['n_steps']
    epoch = checkpoint_dict['epoch']

    return model, optimizer, scheduler, epoch, steps

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler, epoch: int, n_steps: int, checkpoint_path: str):
    checkpoint_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'n_steps': n_steps,
        'epoch': epoch
    }

    torch.save(checkpoint_dict, checkpoint_path)

def load_weights(module: nn.Module, weights: OrderedDict):
    return module.load_state_dict(weights)