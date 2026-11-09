"""
Shared helpers: load the trained model and predict on images.
Used by evaluate.py, predict.py and app.py.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fabric_model.pth"


def load_model(model_path=DEFAULT_MODEL_PATH, device="cpu"):
    """Load the trained MobileNetV2 and its settings (classes, size, mean, std)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, ckpt


def preprocess(img_bgr: np.ndarray, ckpt) -> torch.Tensor:
    """OpenCV BGR image (any size) -> normalised 1x3xHxW tensor."""
    size = ckpt["img_size"]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - np.array(ckpt["mean"], dtype=np.float32)) / np.array(ckpt["std"], dtype=np.float32)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def predict(model, ckpt, img_bgr: np.ndarray, device="cpu"):
    """Return list of (class, probability) sorted from most to least likely."""
    x = preprocess(img_bgr, ckpt).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    order = probs.argsort()[::-1]
    return [(ckpt["classes"][i], float(probs[i])) for i in order]


def read_image(path) -> np.ndarray:
    """Read an image with OpenCV; big phone photos are decoded at reduced size."""
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img
