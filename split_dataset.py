from pathlib import Path
import random 
import shutil

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"

SELECTED_CLASSES = [
    "Actinomyces.israeli",
    "Bacteroides.fragilis",
    "Clostridium.perfringens",
    "Escherichia.coli",
    "Listeria.monocytogenes",
]

SEED = 42
SPLIT = (0.70, 0.20, 0.10) # Train, Val, Test
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS