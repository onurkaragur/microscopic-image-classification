import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return model, classes, tfm

