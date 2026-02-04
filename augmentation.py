import os 
import random
from pathlib import Path
import argparse

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def random_flip(img: Image.Image):
    # Flips the image horizontally or vertically
    if random.random() < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    
def random_rotate(img: Image.Image, max_angle=30):
    # Randomly rotates the image between -max_angle and +max_angle
    if random.random() < 0.5:
        angle = random.uniform(-max_angle, max_angle)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False)
    return img

def random_affine_crop_size(img: Image.Image, scale=(0.9, 1.0)):
    # Random zoom-in crop + resize back to original size
    if random.random() < 0.5:
        w, h = img.size
        scale_factor = random.uniform(scale[0], scale[1])
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        left = random.randint(0, max(0, w - new_w))
        top = random.randint(0, max(0, h - new_h))
        cropped = img.crop(left, top, left + new_w, top + new_h)
        return cropped.resize((w,h), resample=Image.BILINEAR)
    
def adjust_brightness_contrast(img: Image.Image, brightness=(0.8, 1.2), contrast=(0.8,1.2), p=0.8):
    # Randomly changes lighting conditions
    if random.random() < p:
        b = random.uniform(brightness[0], brightness[1])
        img = ImageEnhance.Brightness(img).enhance(b)
    if random.random() < p:
        c = random.uniform(contrast[0], contrast[1])
        img = ImageEnhance.Contrast(img).enhance(c)
    return img

