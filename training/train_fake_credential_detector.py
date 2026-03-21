"""
Training script for Fake Credential Detection on Bank Cards (Cartes Bancaires)
using YOLOv5.

Classes:
    0 = real_card        — Legitimate bank card with consistent credentials
    1 = fake_card        — Bank card with forged / tampered credentials


Detection signals used to label fake cards in synthetic data:
  • Wrong PAN format  (not 16 digits, wrong grouping)
  • Expired or impossible date  (month > 12, year < current)
  • Missing or misplaced logo / hologram placeholder
  • Font inconsistency flags (simulated by colour / size variation)
  • CVV printed on front (should never appear on front for real cards)
  • Mismatched card-scheme logo vs BIN prefix
    (e.g. "VISA" written but number starts with 5 → Mastercard range)

Optimised for 4 GB GPU (uses YOLOv5n by default).
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

# ──────────────────────────────────────────────
# 1. DATASET  (PyTorch Dataset + collate)
# ──────────────────────────────────────────────

from torch.utils.data import Dataset


class FakeCredentialDataset(Dataset):
    """
    YOLO-format dataset for bank-card fake-credential detection.

    Label format per line: class x_center y_center width height  (all 0-1)
    Classes : 0=bank_card
    """

    def __init__(self, image_dir, label_dir, img_size=640):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size

        self.image_files = (
            list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.jpeg"))
        )
        print(f"[Dataset] Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        label_path = self.label_dir / (img_path.stem + ".txt")
        boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    if len(vals) >= 5:
                        boxes.append(vals[:5])

        boxes = np.array(boxes) if boxes else np.zeros((0, 5))
        image_tensor = (
            torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        )
        return image_tensor, boxes, str(img_path)


def collate_fn(batch):
    images, labels, paths = [], [], []
    for img, label, path in batch:
        images.append(img)
        labels.append(torch.FloatTensor(label))
        paths.append(path)
    return torch.stack(images, 0), labels, paths


# ──────────────────────────────────────────────
# 2. YOLO CONFIG  (data.yaml)
# ──────────────────────────────────────────────

CLASS_NAMES = ["bank_card"]
NUM_CLASSES = 1


def create_yolo_config(data_yaml_path):
    config = {
        "path": str(Path(data_yaml_path).parent.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"[Config] data.yaml saved to: {data_yaml_path}")
    return data_yaml_path


# ──────────────────────────────────────────────
# 3. DATASET PREPARATION  (train / val split)
# ──────────────────────────────────────────────

def prepare_yolo_dataset(source_images, source_labels, output_dir, split_ratio=0.8):
    """
    Organise images + labels into YOLOv5 folder structure and write data.yaml.

    output_dir/
    ├── train/images/   train/labels/
    ├── val/images/     val/labels/
    └── data.yaml
    """
    output_dir = Path(output_dir)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (output_dir / split / sub).mkdir(parents=True, exist_ok=True)

    source_images = Path(source_images)
    source_labels = Path(source_labels)

    image_files = (
        list(source_images.glob("*.jpg"))
        + list(source_images.glob("*.png"))
        + list(source_images.glob("*.jpeg"))
    )
    np.random.shuffle(image_files)

    split_idx = int(len(image_files) * split_ratio)
    splits = {"train": image_files[:split_idx], "val": image_files[split_idx:]}

    for split_name, files in splits.items():
        for img_path in tqdm(files, desc=f"Copying {split_name}"):
            shutil.copy(img_path, output_dir / split_name / "images" / img_path.name)
            lbl = source_labels / (img_path.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, output_dir / split_name / "labels" / lbl.name)

    print(
        f"[Dataset] {len(splits['train'])} train  |  {len(splits['val'])} val"
    )
    data_yaml = output_dir / "data.yaml"
    create_yolo_config(data_yaml)
    return data_yaml


# ──────────────────────────────────────────────
# 4. SYNTHETIC DATA GENERATOR
# ──────────────────────────────────────────────

# Visa BINs start with 4, Mastercard with 5, Amex with 34/37
CARD_SCHEMES = {
    "VISA":       {"prefix": "4", "color": (0, 0, 180)},
    "MASTERCARD": {"prefix": "5", "color": (180, 0, 0)},
    "AMEX":       {"prefix": "3", "color": (0, 140, 0)},
}


def _random_pan(scheme_name, corrupt=False):
    """Generate a 16-digit PAN (or a corrupted one)."""
    scheme = CARD_SCHEMES[scheme_name]
    prefix = scheme["prefix"]
    if corrupt:
        # Typical forgery patterns
        fault = np.random.choice(["wrong_prefix", "short", "mismatched_scheme"])
        if fault == "wrong_prefix":
            # Visa card number but Mastercard prefix
            prefix = "5" if scheme_name == "VISA" else "4"
        elif fault == "short":
            # 15 digits instead of 16
            digits = prefix + "".join([str(np.random.randint(0, 10)) for _ in range(14)])
            return digits, True
        # mismatched_scheme → keep going with wrong prefix
    digits = prefix + "".join([str(np.random.randint(0, 10)) for _ in range(15)])
    return digits[:16], corrupt


def _format_pan(pan):
    """Group PAN into 4-4-4-4."""
    return " ".join([pan[i:i+4] for i in range(0, len(pan), 4)])


def _random_expiry(corrupt=False):
    """Return MM/YY, optionally corrupted."""
    if corrupt and np.random.rand() < 0.5:
        # Impossible month
        month = np.random.randint(13, 20)
        year = np.random.randint(20, 30)
    elif corrupt:
        # Past date
        month = np.random.randint(1, 13)
        year = np.random.randint(10, 20)   # already expired
    else:
        month = np.random.randint(1, 13)
        year = np.random.randint(25, 31)   # 2025-2030
    return f"{month:02d}/{year:02d}"


def _draw_card(img, x1, y1, card_w, card_h, scheme_name, is_fake):
    """
    Draw a synthetic bank card on `img`.
    Returns empty list (suspicious_zone class removed).
    """
    img_h, img_w = img.shape[:2]

    scheme = CARD_SCHEMES[scheme_name]
    logo_color = scheme["color"]

    # Card body
    card_color = tuple(int(c) for c in np.random.randint(30, 220, 3))
    cv2.rectangle(img, (x1, y1), (x1 + card_w, y1 + card_h), card_color, -1)
    cv2.rectangle(img, (x1, y1), (x1 + card_w, y1 + card_h), (30, 30, 30), 2)

    # ── Chip placeholder ──────────────────────────────────────────────────
    chip_x, chip_y = x1 + 15, y1 + card_h // 4
    chip_w, chip_h = 40, 30
    if is_fake and np.random.rand() < 0.4:
        # Missing chip → suspicious zone
        pass
    else:
        cv2.rectangle(img, (chip_x, chip_y),
                      (chip_x + chip_w, chip_y + chip_h), (200, 180, 0), -1)
        cv2.rectangle(img, (chip_x, chip_y),
                      (chip_x + chip_w, chip_y + chip_h), (100, 90, 0), 1)

    # ── Scheme logo ───────────────────────────────────────────────────────
    logo_x = x1 + card_w - 80
    logo_y = y1 + 10
    logo_display = scheme_name
    # Fake: wrong logo vs PAN prefix mismatch
    if is_fake and np.random.rand() < 0.5:
        other = [s for s in CARD_SCHEMES if s != scheme_name]
        logo_display = np.random.choice(other)          # mismatch!
        logo_color = CARD_SCHEMES[logo_display]["color"]

    cv2.putText(img, logo_display, (logo_x, logo_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, logo_color, 2)

    # ── PAN ───────────────────────────────────────────────────────────────
    pan_raw, pan_corrupt = _random_pan(scheme_name, corrupt=is_fake and np.random.rand() < 0.7)
    pan_str = _format_pan(pan_raw)
    pan_x = x1 + 10
    pan_y = y1 + card_h // 2 + 10
    pan_color = (0, 0, 0)
    if pan_corrupt:
        # Draw in a slightly different font size to simulate forgery
        font_scale = 0.45 if np.random.rand() < 0.5 else 0.65   # inconsistent
        pan_color = (200, 0, 0)   # red tint → anomaly
    else:
        font_scale = 0.55

    cv2.putText(img, pan_str, (pan_x, pan_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, pan_color, 1)

    # ── Expiry ────────────────────────────────────────────────────────────
    expiry = _random_expiry(corrupt=is_fake and np.random.rand() < 0.5)
    exp_x = x1 + 10
    exp_y = pan_y + 22
    exp_corrupt = (is_fake and ("/" not in expiry or int(expiry[:2]) > 12))
    exp_color = (200, 0, 0) if exp_corrupt else (0, 0, 0)
    cv2.putText(img, f"EXP: {expiry}", (exp_x, exp_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, exp_color, 1)

    # ── CVV on front (should NEVER appear on front of a real card) ────────
    if is_fake and np.random.rand() < 0.3:
        cvv_x = x1 + card_w - 50
        cvv_y = exp_y
        cvv = str(np.random.randint(100, 1000))
        cv2.putText(img, f"CVV:{cvv}", (cvv_x, cvv_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1)

    # ── Hologram placeholder ──────────────────────────────────────────────
    holo_x = x1 + card_w - 40
    holo_y = y1 + card_h - 30
    if is_fake and np.random.rand() < 0.35:
        pass  # Missing hologram
    else:
        cv2.circle(img, (holo_x, holo_y), 12, (0, 200, 200), -1)
        cv2.putText(img, "H", (holo_x - 5, holo_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return []


def create_synthetic_dataset(output_dir, num_images=300):
    """
    Generate synthetic bank-card images with YOLO labels.

    Classes produced:
        0 = bank_card (all cards)
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Synthetic] Creating {num_images} images in {output_dir} …")

    IMG_SIZE = 640

    for i in tqdm(range(num_images)):
        img = np.random.randint(180, 240, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # Light desk / table texture
        noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)

        labels = []
        num_cards = np.random.randint(1, 3)   # 1 or 2 cards per image

        for _ in range(num_cards):
            # Card dimensions (standard aspect ratio ~1.586)
            card_w = np.random.randint(220, 360)
            card_h = int(card_w / 1.586)

            if IMG_SIZE - card_w < 1 or IMG_SIZE - card_h < 1:
                continue

            x1 = np.random.randint(0, IMG_SIZE - card_w)
            y1 = np.random.randint(0, IMG_SIZE - card_h)

            scheme_name = np.random.choice(list(CARD_SCHEMES.keys()))

            # 50% chance of fake card
            is_fake = np.random.rand() < 0.5

            suspicious_boxes = _draw_card(img, x1, y1, card_w, card_h,
                                          scheme_name, is_fake)

            # Card bounding box (normalised)
            x_c = (x1 + card_w / 2) / IMG_SIZE
            y_c = (y1 + card_h / 2) / IMG_SIZE
            w_n = card_w / IMG_SIZE
            h_n = card_h / IMG_SIZE

            # All cards get label 0 regardless of real/fake
            labels.append([0, x_c, y_c, w_n, h_n])

        # Save
        img_path = images_dir / f"card_{i:05d}.jpg"
        cv2.imwrite(str(img_path), img)

        lbl_path = labels_dir / f"card_{i:05d}.txt"
        with open(lbl_path, "w") as f:
            for lbl in labels:
                cls, xc, yc, w, h = lbl
                f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print(f"[Synthetic] Done → {images_dir}")
    return images_dir, labels_dir


# ──────────────────────────────────────────────
# 5. YOLOV5 TRAINING LAUNCHER
# ──────────────────────────────────────────────

def train_yolov5(data_yaml, output_dir, epochs=60, batch_size=8,
                 img_size=640, model_size="n", device="cuda"):
    """
    Clone YOLOv5 (if needed) and launch training via subprocess.

    model_size:
        'n' → nano   (best for ≤4 GB GPU)
        's' → small
        'm' → medium
    """
    print("\n" + "=" * 60)
    print("  YOLOv5 — Fake Credential Detector Training")
    print("=" * 60)
    print(f"  Model   : YOLOv5{model_size}")
    print(f"  Epochs  : {epochs}")
    print(f"  Batch   : {batch_size}")
    print(f"  Img size: {img_size}")
    print(f"  Device  : {device}")
    print(f"  Classes : {CLASS_NAMES}")
    print("=" * 60)

    yolov5_dir = Path(__file__).parent / "yolov5"
    if not yolov5_dir.exists():
        print("\n[Setup] Cloning YOLOv5 …")
        os.system("git clone https://github.com/ultralytics/yolov5.git training/yolov5")
        os.system("pip install -r training/yolov5/requirements.txt")

    data_yaml_abs = Path(data_yaml).absolute()
    output_dir_abs = Path(output_dir).absolute()
    original_dir = os.getcwd()

    try:
        os.chdir(yolov5_dir)

        cmd = [
            sys.executable, "train.py",
            "--img",     str(img_size),
            "--batch",   str(batch_size),
            "--epochs",  str(epochs),
            "--data",    str(data_yaml_abs),
            "--weights", f"yolov5{model_size}.pt",
            "--project", str(output_dir_abs),
            "--name",    "fake_credential_detector",
            "--device",  "0" if device == "cuda" else "cpu",
            "--cache",   "disk",
            "--optimizer", "AdamW",
            "--patience", "15",
            "--exist-ok",
            # Extra regularisation to reduce false positives on real cards
            "--label-smoothing", "0.1",
        ]

        print(f"\n[Train] Command:\n  {' '.join(cmd)}\n")

        if not data_yaml_abs.exists():
            print(f"[ERROR] data.yaml not found: {data_yaml_abs}")
            return None

        result = subprocess.run(cmd)
        os.chdir(original_dir)

        if result.returncode == 0:
            best = output_dir_abs / "fake_credential_detector" / "weights" / "best.pt"
            if best.exists():
                print(f"\n[Done] Best model → {best}")
                return best
            print("[Warning] best.pt not found after training.")
            return None
        else:
            print("[ERROR] Training failed — check logs above.")
            return None

    except Exception as exc:
        os.chdir(original_dir)
        print(f"[Exception] {exc}")
        return None


# ──────────────────────────────────────────────
# 6. INFERENCE HELPER  (quick test after training)
# ──────────────────────────────────────────────

def run_inference(model_path, image_path, conf_threshold=0.35):
    """
    Quick single-image inference using the trained model via YOLOv5 detect.py.
    Saves annotated result in the same folder as the model.
    """
    yolov5_dir = Path(__file__).parent / "yolov5"
    if not yolov5_dir.exists():
        print("[Inference] YOLOv5 directory not found — train first.")
        return

    cmd = [
        sys.executable,
        str(yolov5_dir / "detect.py"),
        "--weights", str(model_path),
        "--source",  str(image_path),
        "--conf",    str(conf_threshold),
        "--project", str(Path(model_path).parent.parent / "inference"),
        "--name",    "results",
        "--exist-ok",
        "--save-txt",
    ]
    print(f"[Inference] {' '.join(cmd)}")
    subprocess.run(cmd)


# ──────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train YOLOv5 to detect fake credentials on bank cards (cartes bancaires)"
    )
    parser.add_argument("--images",  default="training/datasets/fake_cards/images",
                        help="Directory of training images")
    parser.add_argument("--labels",  default="training/datasets/fake_cards/labels",
                        help="Directory of YOLO-format label .txt files")
    parser.add_argument("--output",  default="training/runs",
                        help="Output directory for trained weights")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch-size",  type=int,   default=8,
                        help="Reduce to 4 for ≤4 GB GPU")
    parser.add_argument("--img-size",    type=int,   default=640)
    parser.add_argument("--model-size",  default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="'n' (nano) recommended for 4 GB GPU")
    parser.add_argument("--synthetic",       action="store_true",
                        help="Generate synthetic dataset before training")
    parser.add_argument("--num-synthetic",   type=int, default=400,
                        help="Number of synthetic images to generate")
    parser.add_argument("--infer",  default=None,
                        help="Run inference on this image after training")

    args = parser.parse_args()

    output_dir = Path(args.output).absolute()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    # ── Synthetic data ──────────────────────────────────────────────
    if args.synthetic or not images_dir.exists():
        print("[Main] Generating synthetic dataset …")
        images_dir, labels_dir = create_synthetic_dataset(
            images_dir.parent,
            num_images=args.num_synthetic,
        )

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not image_files:
        print("[Error] No images found. Use --synthetic to auto-generate data.")
        return

    print(f"\n[Main] {len(image_files)} images found.")

    # ── Dataset preparation ─────────────────────────────────────────
    yolo_dataset_dir = output_dir / "dataset"
    data_yaml = prepare_yolo_dataset(images_dir, labels_dir,
                                     yolo_dataset_dir, split_ratio=0.8)

    # ── Device ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Main] Device: {device}")
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"       GPU  : {name}  ({mem:.1f} GB)")
        if mem < 5:
            print("       ⚠️  <5 GB GPU — use --batch-size 4 --model-size n")

    # ── Train ───────────────────────────────────────────────────────
    best_model = train_yolov5(
        data_yaml   = data_yaml,
        output_dir  = args.output,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        img_size    = args.img_size,
        model_size  = args.model_size,
        device      = device,
    )

    if best_model:
        print("\n" + "=" * 60)
        print("  Training complete!")
        print(f"  Best weights : {best_model}")
        print("\n  Usage:")
        print(f"    python detect.py --weights {best_model} \\")
        print( "      --source path/to/card_image.jpg --conf 0.35")
        print("=" * 60)

        # Optional inference
        if args.infer:
            run_inference(best_model, args.infer)
    else:
        print("\n[Main] Training did not complete successfully.")
        print("  Common fixes:")
        print("    • Out of memory  → --batch-size 4")
        print("    • Missing deps   → pip install -r training/yolov5/requirements.txt")


if __name__ == "__main__":
    main()