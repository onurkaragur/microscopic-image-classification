import os 
import random
from pathlib import Path
import argparse

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def random_flip(img: Image.Image, p=0.5):
    # Flips the image horizontally or vertically
    if random.random() < p:
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img
    
def random_rotate(img: Image.Image, max_angle=30, p=0.5):
    # Randomly rotates the image between -max_angle and +max_angle
    if random.random() < p:
        angle = random.uniform(-max_angle, max_angle)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False)
    return img

def random_affine_crop_resize(img: Image.Image, scale=(0.9, 1.0), p=0.5):
    # Random zoom-in crop + resize back to original size
    if random.random() < p:
        w, h = img.size
        scale_factor = random.uniform(scale[0], scale[1])
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        left = random.randint(0, max(0, w - new_w))
        top = random.randint(0, max(0, h - new_h))
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w,h), resample=Image.BILINEAR)
    return img
    
def adjust_brightness_contrast(img: Image.Image, brightness=(0.8, 1.2), contrast=(0.8,1.2), p=0.8):
    # Randomly changes lighting conditions
    if random.random() < p:
        b = random.uniform(brightness[0], brightness[1])
        img = ImageEnhance.Brightness(img).enhance(b)
    if random.random() < p:
        c = random.uniform(contrast[0], contrast[1])
        img = ImageEnhance.Contrast(img).enhance(c)
    return img

def add_gaussian_noise(img: Image.Image, mean=0.0, std=5.0, p=0.5):
    # Adds pixel noise
    if random.random() < p:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(mean, std, arr.shape)
        arr += noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    return img

def random_blur(img: Image.Image, p=0.3):
    # Adds blur to a random radius
    if random.random() < p:
        radius = random.uniform(0.3, 1.5)
        return img.filter(ImageFilter.GaussianBlur(radius))
    return img

# Main augmentation pipeline
def augment_image(img: Image.Image):
    # Compose a few random operations
    img = random_flip(img, p=0.5)
    img = random_rotate(img, max_angle=30, p=0.7)
    img = random_affine_crop_resize(img, scale=(0.9, 1.0), p=0.6)
    img = adjust_brightness_contrast(img, p=0.9)
    img = random_blur(img, p=0.3)
    img = add_gaussian_noise(img, std=6.0, p=0.4)
    return img

def is_image_file(path: Path):
    return path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

def augment_directory(src: str, dst: str, per_image: int = 5, seed: int = None):
    """
    Create augmented images for each image found under "src" (supports class subfolders).

    Args:
        src: source directory containing images or class subdirectories
        dst: target directory where augmented images will be saved
        per_image: number of augmented samples to create per original image
        seed: random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    # Walk source tree
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst.joinpath(rel)
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            p = Path(root) / f
            if not is_image_file(p):
                continue
            try:
                with Image.open(p) as img:
                    img = img.convert('RGB')
                    base_name = p.stem
                    # Save a copy of original if not present in dst
                    orig_out = target_dir / f
                    if not orig_out.exists():
                        img.save(orig_out)
                    for i in range(per_image):
                        aug = augment_image(img)
                        out_name = f"{base_name}_aug{i+1}{p.suffix}"
                        out_path = target_dir / out_name
                        aug.save(out_path)
            except Exception as e:
                print(f"Skipping {p} due to error: {e}")

def create_preview(src: str, dst: str, samples_per_class: int = 1, seed: int = None):
    """Create a small preview grid of augmented images for each class.

    This picks up to `samples_per_class` images per class, augments them, and writes thumbnails to `dst`.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # If src contains class directories, iterate classes
    for class_dir in sorted([d for d in src.iterdir() if d.is_dir()]):
        images = [p for p in class_dir.iterdir() if p.is_file() and is_image_file(p)]
        if not images:
            continue
        chosen = images[:samples_per_class]
        for idx, p in enumerate(chosen):
            try:
                with Image.open(p) as img:
                    img = img.convert('RGB')
                    aug = augment_image(img)
                    thumb = aug.copy()
                    thumb.thumbnail((256, 256))
                    out_name = f"{class_dir.name}_sample{idx+1}.jpg"
                    thumb.save(dst / out_name)
            except Exception as e:
                print(f"Preview skip {p}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description='Dataset augmentation utility')
    p.add_argument('--src', required=True, help='Source dataset directory')
    p.add_argument('--dst', required=True, help='Destination directory for augmented images')
    p.add_argument('--per-image', type=int, default=5, help='Augmented images per original')
    p.add_argument('--seed', type=int, default=None, help='Random seed')
    p.add_argument('--preview', action='store_true', help='Create small preview thumbnails instead of full augmentation')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.preview:
        print(f"Creating preview thumbnails from {args.src} into {args.dst}")
        create_preview(args.src, args.dst, samples_per_class=1, seed=args.seed)
    else:
        print(f"Augmenting dataset from {args.src} into {args.dst} ({args.per_image} per image)")
        augment_directory(args.src, args.dst, per_image=args.per_image, seed=args.seed)




