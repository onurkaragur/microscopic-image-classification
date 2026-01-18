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

def main():
    cfg = Config()
    os.makedirs("models", exist_ok=True)

    train_tfms = transforms.Compose([
        transforms.Resize(cfg.img_size, cfg.img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(cfg.train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(cfg.val_dir, transform=val_tfms)

    num_classes = len(train_ds.classes)
    print("Classes: ", train_ds.classes)
    assert num_classes == 8, f"Found 8 classes as expected: {num_classes}"

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers) 

    device = get_device()
    print("Device: ", device)

    # Transfer Learning
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val_acc = 0.0
    