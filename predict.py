import argparse
from pathlib import Path

import cv2
import numpy as np
import torch 
import torch.nn as nn
from torchvision import models, transforms

def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size",224)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, classes, img_size