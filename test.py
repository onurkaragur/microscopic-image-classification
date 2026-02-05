import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Configuration
    test_dir = "data/processed/test"
    model_path = "models/best_model.pth"
    img_size = 224
    batch_size = 16

    # Load model
    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(model_path, map_location=device)
    classes = ckpt["classes"]
    num_classes = len(classes)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model with {num_classes} classes: {classes}")

    # Test transforms
    test_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load test dataset
    test_ds = datasets.ImageFolder(test_dir, transform=test_tfms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Test dataset: {len(test_ds)} samples")

    # Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    accuracy = correct / total
    print(f"Test accuracy: {accuracy}")
    print(".4f")

if __name__ == "__main__":
    main()