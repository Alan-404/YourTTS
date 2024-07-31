import torch
from torch.utils.data import Dataset, Sampler, BatchSampler
import pandas as pd
from processing.processor import YourTTSProcessor
from tqdm import tqdm
from typing import Any, Optional, Tuple
import itertools
import random

class YourTTSDataset(Dataset):
    def __init__(self, manifest_path: str, processor: YourTTSProcessor) -> None:
        super().__init__()