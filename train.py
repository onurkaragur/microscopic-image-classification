import os
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

@dataclass
class Config:
    train_dir: str = "data/processed/train"
    val_dir: str = "data/processed/val"
    out_path: str = "models"
    img_size: int = 224
    batch_size: int = 16
    lr: float = 1e-4
    epochs: int = 8
    num_workers: int = 0 

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
 