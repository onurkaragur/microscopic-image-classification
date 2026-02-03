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
    
