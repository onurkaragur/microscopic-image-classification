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
        transforms.Resize((cfg.img_size, cfg.img_size)),
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

    for epoch in range(1, cfg.epochs + 1):
        # Training
        model.train()
        train_correct, train_total, train_loss_sum = 0, 0, 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += x.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)

        val_acc = val_correct / max(val_total, 1)

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}\n")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "img_size": cfg.img_size,
            }
            ckpt_path = os.path.join(cfg.out_path, "best_model.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved best model -> {ckpt_path} (val_acc={best_val_acc:.3f})")

        print(f"Done. Best val_acc: {best_val_acc}")

if __name__ == "__main__":
    main()
