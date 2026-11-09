"""
Phase 5 - Train the fabric defect classifier (MobileNetV2 transfer learning).

Two training stages:
  Stage 1: freeze the pre-trained MobileNetV2 body, train only the new
           5-class output layer (fast, learns the basics).
  Stage 2: unfreeze everything and fine-tune with a small learning rate
           (adapts the model to fabric textures).

The best model (highest validation accuracy) is saved to models/fabric_model.pth
Training curves are saved to results/training_curves.png

Run from the project root:
    python src/train.py

Works on CPU (laptop) and on GPU (e.g. Google Colab) without changes.
"""

import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

# ---------------- Settings ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "fabric_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASSES = ["good", "hole", "stain", "missing_yarn", "needle_mark"]
IMG_SIZE = 224
BATCH_SIZE = 32
STAGE1_EPOCHS = 5        # head only
STAGE2_EPOCHS = 10       # full fine-tune
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
SEED = 42

# ImageNet colour statistics (MobileNetV2 was trained with these)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# ------------------------------------------


class FabricDataset(Dataset):
    """Loads all images of one split into memory using OpenCV."""

    def __init__(self, split: str, transform):
        self.transform = transform
        self.images, self.labels = [], []
        for label, cls in enumerate(CLASSES):
            for p in sorted((DATA_DIR / split / cls).glob("*.jpg")):
                img = cv2.imread(str(p))                     # BGR, uint8
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # -> RGB
                if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                self.images.append(img)
                self.labels.append(label)
        if not self.images:
            raise RuntimeError(f"No images found in {DATA_DIR / split}. Run prepare_data.py first.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # HWC uint8 numpy -> CHW uint8 tensor
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1)
        return self.transform(img), self.labels[idx]


def get_transforms():
    train_tf = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(15),
        v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ])
    eval_tf = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def build_model(num_classes: int):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    # Replace the last layer: 1000 ImageNet classes -> our 5 fabric classes
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def set_backbone_trainable(model, trainable: bool):
    for p in model.features.parameters():
        p.requires_grad = trainable


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def plot_history(history, stage1_epochs):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in [(ax1, "acc", "Accuracy"), (ax2, "loss", "Loss")]:
        ax.plot(epochs, history[f"train_{key}"], "o-", label="train")
        ax.plot(epochs, history[f"val_{key}"], "o-", label="validation")
        ax.axvline(stage1_epochs + 0.5, color="gray", ls="--", lw=1)
        ax.text(stage1_epochs + 0.6, ax.get_ylim()[1], " fine-tune →", va="top", color="gray", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=120)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_tf, eval_tf = get_transforms()
    print("Loading images...", flush=True)
    train_ds = FabricDataset("train", train_tf)
    val_ds = FabricDataset("val", eval_tf)
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = build_model(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_acc, best_epoch = 0.0, 0
    MODEL_PATH.parent.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    stages = [
        ("Stage 1: train new head (backbone frozen)", False, STAGE1_EPOCHS, STAGE1_LR),
        ("Stage 2: fine-tune whole model", True, STAGE2_EPOCHS, STAGE2_LR),
    ]
    epoch = 0
    start = time.time()
    for name, unfreeze, n_epochs, lr in stages:
        print(f"\n=== {name} ===", flush=True)
        set_backbone_trainable(model, unfreeze)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
            scheduler.step()

            for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc]):
                history[k].append(v)

            marker = ""
            if va_acc > best_acc:
                best_acc, best_epoch = va_acc, epoch
                torch.save({"model_state": model.state_dict(),
                            "classes": CLASSES,
                            "img_size": IMG_SIZE,
                            "mean": MEAN, "std": STD,
                            "val_acc": va_acc, "epoch": epoch}, MODEL_PATH)
                marker = "  <- best, saved"
            print(f"Epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.1%} | "
                  f"val loss {va_loss:.3f} acc {va_acc:.1%} | {time.time() - t0:.0f}s{marker}",
                  flush=True)

    minutes = (time.time() - start) / 60
    print(f"\nDone in {minutes:.1f} min. Best val accuracy {best_acc:.1%} at epoch {best_epoch}.")
    print(f"Model saved to {MODEL_PATH}")

    plot_history(history, STAGE1_EPOCHS)
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump({**history, "best_val_acc": best_acc, "best_epoch": best_epoch,
                   "minutes": round(minutes, 1)}, f, indent=2)
    print(f"Curves saved to {RESULTS_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    main()
