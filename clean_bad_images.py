from pathlib import Path
from PIL import Image

DATA_DIR = Path(__file__).resolve.parent / "data" / "processed"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def check_and_remove(root: Path):
    removed = 0
    total = 0
    for p in root.rglob("*"):
        if p.is_file() and is_image(p):
            total += 1
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                print("DELETED (Damaged/Unreadable): ", p)
                try:
                    p.unlink()
                    removed += 1
                except Exception as e:
                    print(f"Couldn't delete {p} \n Error: {e}")
    print("Finished. Controlled Total {total}, Deleted {removed}.")

    