"""
Phase 6 - Evaluate the trained model on the TEST set (images never seen in training).

Outputs:
  - accuracy, precision, recall, F1 per class (printed + results/test_report.txt)
  - results/confusion_matrix.png
  - results/wrong_predictions.png  (examples the model got wrong)

Run from the project root:
    python src/evaluate.py
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model_utils import PROJECT_ROOT, load_model, predict

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "test"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    model, ckpt = load_model()
    classes = ckpt["classes"]
    print(f"Loaded model (best val acc {ckpt['val_acc']:.1%}, epoch {ckpt['epoch']})")

    y_true, y_pred, wrong = [], [], []
    for label, cls in enumerate(classes):
        for p in sorted((DATA_DIR / cls).glob("*.jpg")):
            img = cv2.imread(str(p))
            top_cls, top_prob = predict(model, ckpt, img)[0]
            y_true.append(label)
            y_pred.append(classes.index(top_cls))
            if top_cls != cls:
                wrong.append((p, cls, top_cls, top_prob))

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    text = f"Test images: {len(y_true)}\nTest accuracy: {acc:.1%}\n\n{report}"
    print("\n" + text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "test_report.txt").write_text(text, encoding="utf-8")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix – test accuracy {acc:.1%}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # Examples of mistakes
    if wrong:
        n = min(10, len(wrong))
        fig, axes = plt.subplots(2, 5, figsize=(14, 6.5))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (p, true_cls, pred_cls, prob) in zip(axes.flat, wrong[:n]):
            ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
            ax.set_title(f"actual: {true_cls}\npredicted: {pred_cls} ({prob:.0%})", fontsize=9)
        fig.suptitle(f"Wrong predictions ({len(wrong)} of {len(y_true)})")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "wrong_predictions.png", dpi=110)
        plt.close(fig)

    print(f"Saved report and charts to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
