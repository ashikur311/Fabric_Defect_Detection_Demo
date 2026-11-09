"""
Predict the defect type of one image or every image in a folder.

Examples (run from the project root):
    python src/predict.py path\\to\\photo.jpg
    python src/predict.py data\\my_denim_samples

For a folder, a picture grid is also saved to results/my_denim_predictions.png
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_utils import PROJECT_ROOT, load_model, predict, read_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:
        files = [target]
    if not files:
        print(f"No images found in {target}")
        sys.exit(1)

    model, ckpt = load_model()
    results = []
    for p in files:
        img = read_image(p)
        preds = predict(model, ckpt, img)
        top_cls, top_prob = preds[0]
        runner_up = ", ".join(f"{c} {pr:.0%}" for c, pr in preds[1:3])
        print(f"{p.name:<40} -> {top_cls:<13} {top_prob:6.1%}   (next: {runner_up})")
        results.append((img, p.name, top_cls, top_prob))

    if len(results) > 1:
        cols = 5
        rows = (len(results) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.1), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (img, name, cls, prob) in zip(axes.flat, results):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{cls} ({prob:.0%})\n{name[:24]}", fontsize=8)
        fig.tight_layout()
        out = PROJECT_ROOT / "results" / "my_denim_predictions.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"\nPicture grid saved to {out}")


if __name__ == "__main__":
    main()
