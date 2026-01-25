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

def predict(model, classes, tfm, device, pil_img):
    img = img = pil_img.convert("RGB")


    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        temperature = 1.5
        logits = model(x) / temperature
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        idxs = probs.argsort()[::-1]
        top1, top2 = int(idxs[0]), int(idxs[1])
        return classes[top1], float(probs[top1]), classes[top2], float(probs[top2])
    
def preprocess_clahe(pil_img):
    # PIL -> numpy
    img = np.array(pil_img.convert("RGB"))

    # RGB -> LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    # LAB -> RGB
    lab2 = cv2.merge((l2, a, b))
    rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    return Image.fromarray(rgb2)

def draw_label(pil_img, text):
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    x, y = 15, 15
    pad = 10
    tw, th = draw.textsize(text, font=font)

    draw.rectangle(
        [x - pad, y - pad, x + tw + pad, y + th + pad],
        fill=(0, 0, 0)
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return img

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bakteri Sınıflandırma")
        self.geometry("900x650")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "models/bacteria_resnet18.pt"





        if not Path(self.model_path).exists():
            messagebox.showerror("Hata", "Model bulunamadı!")
            self.destroy()
            return

        self.model, self.classes, self.tfm = load_model(self.model_path, self.device)

        # ---- UI ----
        top = tk.Frame(self)
        top.pack(pady=10)

        self.btn_select = tk.Button(top, text="Resim Seç", width=15, command=self.select_image)
        self.btn_select.pack(side="left", padx=10)

        self.btn_predict = tk.Button(top, text="Tahmin Et", width=15,
                                     command=self.run_predict, state="disabled")
        self.btn_predict.pack(side="left", padx=10)

        self.info = tk.StringVar(value="Tahmin: -")
        tk.Label(self, textvariable=self.info, font=("Arial", 14)).pack(pady=5)

        self.image_box = tk.Label(self, bd=2, relief="groove")
        self.image_box.pack(expand=True, fill="both", padx=10, pady=10)

        self.current_pil = None
        self.current_tk = None

        self.out_dir = Path("outputs")
        self.out_dir.mkdir(exist_ok=True)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Resim Seç",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return

        try:
            img = Image.open(path)
        except Exception as e:
            messagebox.showerror("Hata", str(e))
            return

        self.current_pil = img
        self.btn_predict.config(state="normal")
        self.info.set("Tahmin: -")
        self.show_image(img)

    def show_image(self, pil_img):
        w = self.image_box.winfo_width() or 850
        h = self.image_box.winfo_height() or 520

        img = pil_img.copy()
        img.thumbnail((w - 20, h - 20))

        self.current_tk = ImageTk.PhotoImage(img)
        self.image_box.config(image=self.current_tk)

    def run_predict(self):
        if self.current_pil is None:
            return

        name1, conf1, name2, conf2 = predict(
            self.model, self.classes, self.tfm, self.device, self.current_pil
        )

        THR = 0.50  # top-1 güven eşiği
        MARGIN = 0.10  # top1 - top2 fark eşiği

        if (conf1 < THR) or ((conf1 - conf2) < MARGIN):
            text = "Unknown"
        else:
            text = f"{name1} ({conf1 * 100:.1f}%)"

        self.info.set("Tahmin: " + text)

        labeled = draw_label(self.current_pil, text)
        labeled.save(self.out_dir / "sonuc_gui.jpg")
        self.show_image(labeled)


if __name__ == "__main__":
    App().mainloop()