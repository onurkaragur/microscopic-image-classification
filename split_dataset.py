from pathlib import Path
import random 
import shutil

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"

SELECTED_CLASSES = [
    "Amoeba",
    "Euglena",
    "Hydra",
    "Paramecium",
    "Rod_bacteria",
    "Spherical_bacteria",
    "Spiral_bacteria",
    "Yeast"
]

SEED = 42
SPLIT = (0.70, 0.20, 0.10) # Train, Val, Test
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def reset_data_folders():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    (OUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "val").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "test").mkdir(parents=True, exist_ok=True)

def copy_files(files: list[Path], dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files: 
        shutil.copy(f, dst_dir / f.name)

def main():

    random.seed(SEED)

    # Checks if selected classes exist
    for cls in SELECTED_CLASSES:
        p = RAW_DIR / cls
        print(cls, "-> ", "OK" if p.exists() else "None")
        if not p.exists():
            raise FileNotFoundError(f"Couldn't find class folder: {p}")
        
    reset_data_folders()
    print("Data folders are ready!")

    # Dataset split
    for cls in SELECTED_CLASSES:
        cls_dir = RAW_DIR / cls
        imgs = [p for p in cls_dir.rglob("*") if p.is_file() and is_image(p)]
        if len(imgs) == 0:
            raise RuntimeError(f"Couldn't find any images inside {cls}.")

        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * SPLIT[0])
        n_val = int(n * SPLIT[1])

        train_files = imgs[:n_train]
        val_files = imgs[n_train: n_train + n_val]
        test_files = imgs[n_train + n_val:]

        copy_files(train_files, OUT_DIR / "train" / cls)
        copy_files(val_files, OUT_DIR / "val" / cls)
        copy_files(test_files, OUT_DIR / "test" / cls)

        print(f"[{cls}] total={n}  train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")  

    print("Split operation completed.")  

if __name__ == "__main__":
    main() 