"""
Phase 4 - Explore the prepared dataset.

Creates two charts in results/:
  - class_distribution.png : how many images per class in train/val/test
  - sample_images.png      : 5 example images from each class

Run from the project root (after prepare_data.py):
    python src/explore_data.py
"""

import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")               # save charts to file, no pop-up window
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASSES = ["good", "hole", "stain", "missing_yarn", "needle_mark"]
SPLITS = ["train", "val", "test"]
SAMPLES_PER_CLASS = 5


def count_images():
    return {s: [len(list((DATA_DIR / s / c).glob("*.jpg"))) for c in CLASSES]
            for s in SPLITS}


def plot_distribution(counts):
    x = np.arange(len(CLASSES))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, s in enumerate(SPLITS):
        bars = ax.bar(x + (i - 1) * width, counts[s], width, label=s)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Number of images")
    ax.set_title("Images per class (train / val / test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)


def plot_samples():
    rng = random.Random(0)
    fig, axes = plt.subplots(len(CLASSES), SAMPLES_PER_CLASS,
                             figsize=(SAMPLES_PER_CLASS * 2.2, len(CLASSES) * 2.3))
    for r, cls in enumerate(CLASSES):
        files = sorted((DATA_DIR / "train" / cls).glob("*.jpg"))
        picks = rng.sample(files, min(SAMPLES_PER_CLASS, len(files)))
        for c in range(SAMPLES_PER_CLASS):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                img = cv2.imread(str(picks[c]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # OpenCV is BGR
                ax.imshow(img)
            if c == 0:
                ax.set_title(cls, loc="left", fontsize=11, fontweight="bold")
    fig.suptitle("Sample training images per class", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_images.png", dpi=120)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    counts = count_images()
    print(f"{'class':<14}" + "".join(f"{s:>8}" for s in SPLITS))
    for i, cls in enumerate(CLASSES):
        print(f"{cls:<14}" + "".join(f"{counts[s][i]:>8}" for s in SPLITS))
    plot_distribution(counts)
    plot_samples()
    print(f"\nCharts saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
