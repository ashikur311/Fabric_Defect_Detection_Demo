"""
Phase 4 - Prepare the dataset with OpenCV.

What it does:
1. Reads the raw FabriD images from data/raw/<class folder>/
2. Groups augmented copies of the same photo together (Roboflow export names
   look like IMG_20260714_201846_jpg.rf.<random>.jpg) so copies never end up
   in both train and test - that would make the test score falsely high.
3. Removes photos labelled with two different classes (ambiguous).
4. Splits each class 70% train / 15% val / 15% test (by original photo).
5. Loads each image with OpenCV, resizes to 224x224 and saves it to
   data/processed/<split>/<class>/

Run from the project root:
    python src/prepare_data.py
"""

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2

# ---------------- Settings ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

IMG_SIZE = 224                      # MobileNetV2 input size
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42                           # fixed seed -> same split every run

# raw folder name  ->  clean class name
CLASS_MAP = {
    "defect free": "good",
    "hole Defect": "hole",
    "stain Defect": "stain",
    "missing yarn Defect": "missing_yarn",
    "Needle Mark Defect": "needle_mark",
}
# ------------------------------------------


def original_name(filename: str) -> str:
    """'IMG_123_jpg.rf.abc.jpg' -> 'IMG_123' (groups augmented copies)."""
    return re.sub(r"_jpg\.rf\..*$", "", filename, flags=re.IGNORECASE)


def collect_images():
    """Return {class: {original_name: [paths]}} using top-level files only."""
    data = defaultdict(lambda: defaultdict(list))
    for folder, cls in CLASS_MAP.items():
        folder_path = RAW_DIR / folder
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder_path}")
        # glob("*.jpg") is NOT recursive -> nested duplicate folders are ignored
        for p in sorted(folder_path.glob("*.jpg")):
            data[cls][original_name(p.name)].append(p)
    return data


def remove_ambiguous(data):
    """Drop photos that appear in more than one class."""
    seen = defaultdict(set)
    for cls, groups in data.items():
        for name in groups:
            seen[name].add(cls)
    ambiguous = {n for n, classes in seen.items() if len(classes) > 1}
    for cls in data:
        for name in ambiguous:
            data[cls].pop(name, None)
    return ambiguous


def split_groups(names):
    """Shuffle original photo names and cut into train/val/test."""
    names = sorted(names)
    random.Random(SEED).shuffle(names)
    n = len(names)
    n_train = round(n * SPLITS["train"])
    n_val = round(n * SPLITS["val"])
    return {
        "train": names[:n_train],
        "val": names[n_train:n_train + n_val],
        "test": names[n_train + n_val:],
    }


def clear_processed():
    """Delete old processed images (keeps folders and .gitkeep files)."""
    for p in OUT_DIR.rglob("*.jpg"):
        p.unlink()


def load_and_resize(path: Path):
    """Load with OpenCV and resize to IMG_SIZE x IMG_SIZE."""
    # IMREAD_REDUCED_COLOR_4 decodes at 1/4 size -> much faster for big
    # phone photos (4160x3120 -> 1040x780), still far above 224x224.
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    # INTER_AREA = best quality when shrinking images
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def main():
    print("Reading raw images...")
    data = collect_images()
    ambiguous = remove_ambiguous(data)
    if ambiguous:
        print(f"Removed {len(ambiguous)} photos labelled with 2 classes: {sorted(ambiguous)}")

    clear_processed()
    counts = defaultdict(lambda: defaultdict(int))
    failed = []
    log_rows = []

    for cls, groups in data.items():
        for split, names in split_groups(groups.keys()).items():
            out_folder = OUT_DIR / split / cls
            out_folder.mkdir(parents=True, exist_ok=True)
            for name in names:
                for i, src in enumerate(groups[name]):
                    img = load_and_resize(src)
                    if img is None:
                        failed.append(src)
                        continue
                    out_name = f"{name}_{i}.jpg"
                    cv2.imwrite(str(out_folder / out_name), img,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    counts[split][cls] += 1
                    log_rows.append([split, cls, out_name, src.relative_to(RAW_DIR).as_posix()])
        print(f"  done: {cls}")

    # Save a record of which raw file went where
    with open(OUT_DIR / "split_info.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "processed_file", "raw_file"])
        writer.writerows(log_rows)

    # Summary table
    classes = list(CLASS_MAP.values())
    print("\n" + f"{'class':<14}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'total':>8}")
    for cls in classes:
        row = [counts[s][cls] for s in SPLITS]
        print(f"{cls:<14}" + "".join(f"{v:>8}" for v in row) + f"{sum(row):>8}")
    totals = [sum(counts[s].values()) for s in SPLITS]
    print(f"{'TOTAL':<14}" + "".join(f"{v:>8}" for v in totals) + f"{sum(totals):>8}")

    if failed:
        print(f"\nWARNING: {len(failed)} images could not be read:")
        for p in failed:
            print("  ", p)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
