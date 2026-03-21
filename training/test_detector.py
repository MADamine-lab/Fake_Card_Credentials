"""
test_detector.py
----------------
Run the full fake credential detection pipeline locally (no web server needed).

Pipeline:
    1. YOLO  — locates all bank_card regions (class 0) in the image
    2. OCR   — runs Luhn, scheme check, expiry, CVV-on-front on each crop

Usage:
    python test_detector.py --model path/to/best.pt --image path/to/card.jpg
    python test_detector.py --model path/to/best.pt --folder path/to/images/
    python test_detector.py --model path/to/best.pt --image card.jpg --show
    python test_detector.py --model path/to/best.pt --image card.jpg --save
    python test_detector.py --model path/to/best.pt --image card.jpg --no-ocr

Requirements:
    pip install torch torchvision opencv-python pytesseract
    + Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Tesseract binary path (Windows) ───────────────────────────────────────
# After installing Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
# set the path here if tesseract.exe is not in your system PATH:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

TRAIN_IMG_SIZE = 320

_SCHEME_PREFIXES = {
    "VISA":       ["4"],
    "MASTERCARD": ["51", "52", "53", "54", "55", "2"],
    "AMEX":       ["34", "37"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Luhn algorithm
# ─────────────────────────────────────────────────────────────────────────────

def luhn_check(pan: str) -> bool:
    pan = re.sub(r"\D", "", pan)
    if not (13 <= len(pan) <= 19):
        return False
    digits = [int(d) for d in pan]
    digits.reverse()
    total = 0
    for i, n in enumerate(digits):
        if i % 2 == 1:
            d = n * 2
            total += d - 9 if d > 9 else d
        else:
            total += n
    return total % 10 == 0


# ─────────────────────────────────────────────────────────────────────────────
# OCR rule engine
# ─────────────────────────────────────────────────────────────────────────────

def ocr_validate(crop_bgr):
    """Run OCR on a card crop and apply all 4 rules. Returns (issues, raw_text)."""
    issues = []
    raw_text = ""

    try:
        import pytesseract

        h, w = crop_bgr.shape[:2]
        scale = max(1.0, 300 / min(h, w))
        if scale > 1.0:
            crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)),
                                  interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        thr  = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 8)
        raw_text = pytesseract.image_to_string(thr, config="--psm 6 --oem 3")

        # Rule 1: PAN Luhn check
        pan_matches = re.findall(r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,4})\b", raw_text)
        if pan_matches:
            pan = re.sub(r"\D", "", pan_matches[0])
            if not luhn_check(pan):
                issues.append({
                    "type":    "INVALID_PAN",
                    "message": f"PAN '{pan_matches[0].strip()}' fails Luhn check",
                    "detail":  f"Digits: {pan}",
                })

            # Rule 2: Scheme / BIN prefix consistency
            text_upper = raw_text.upper()
            for scheme, prefixes in _SCHEME_PREFIXES.items():
                if scheme in text_upper:
                    if not any(pan.startswith(p) for p in prefixes):
                        issues.append({
                            "type":    "SCHEME_MISMATCH",
                            "message": f"Logo {scheme} found but BIN '{pan[:6]}' is incompatible",
                            "detail":  f"Expected prefixes: {prefixes}",
                        })
                    break
        else:
            issues.append({
                "type":    "NO_PAN_FOUND",
                "message": "No card number readable in this region",
                "detail":  "OCR could not find a 16-digit PAN",
            })

        # Rule 3: Expiry date
        expiry_matches = re.findall(r"\b(\d{1,2})[\/\-](\d{2,4})\b", raw_text)
        if expiry_matches:
            month = int(expiry_matches[0][0])
            year  = int(expiry_matches[0][1])
            if year < 100:
                year += 2000
            now = datetime.now()
            if not (1 <= month <= 12):
                issues.append({
                    "type":    "INVALID_EXPIRY_MONTH",
                    "message": f"Impossible expiry month: {month:02d}",
                    "detail":  "Month must be 01-12",
                })
            elif year < now.year or (year == now.year and month < now.month):
                issues.append({
                    "type":    "EXPIRED_CARD",
                    "message": f"Card expired: {month:02d}/{year}",
                    "detail":  f"Today is {now.month:02d}/{now.year}",
                })
            elif year > now.year + 10:
                issues.append({
                    "type":    "INVALID_EXPIRY_YEAR",
                    "message": f"Suspicious expiry: {month:02d}/{year}",
                    "detail":  "More than 10 years ahead",
                })

        # Rule 4: CVV on front
        if re.search(r"\bCVV\b[\s:]*\d{3,4}", raw_text, re.IGNORECASE):
            issues.append({
                "type":    "CVV_ON_FRONT",
                "message": "CVV printed on front — never valid on a real card",
                "detail":  "CVV/CVC should only appear on the back",
            })

    except ImportError:
        print("  [WARN] pytesseract not installed — OCR skipped.")
        print("         pip install pytesseract  +  Tesseract binary")
    except Exception as e:
        print(f"  [WARN] OCR error: {e}")

    return issues, raw_text


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path, conf, iou):
    p = Path(model_path)
    if not p.exists():
        print(f"[ERROR] Model not found: {p.absolute()}")
        sys.exit(1)

    print(f"[INFO]  Model    : {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"[INFO]  Device   : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"[INFO]  Conf     : {conf}   IOU: {iou}   Img size: {TRAIN_IMG_SIZE}")

    # Use local YOLOv5 source if available (no internet)
    yolov5_dir = Path("training/yolov5")
    if not yolov5_dir.exists():
        yolov5_dir = p.parent.parent.parent.parent / "training" / "yolov5"

    if yolov5_dir.exists():
        print(f"[INFO]  YOLOv5   : {yolov5_dir} (local)")
        model = torch.hub.load(str(yolov5_dir), "custom", path=str(p),
                               source="local", force_reload=False, verbose=False)
    else:
        print("[INFO]  YOLOv5   : torch.hub (internet)")
        model = torch.hub.load("ultralytics/yolov5", "custom", path=str(p),
                               force_reload=False, verbose=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.conf = conf
    model.iou  = iou
    model.eval()
    print(f"[INFO]  Classes  : {model.names}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Per-image pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_image(model, image_path, use_ocr, show, save):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"  [WARN] Cannot read: {image_path}")
        return 0, 0

    h, w = img_bgr.shape[:2]
    print(f"\n{'━'*60}")
    print(f"  Image : {image_path.name}  ({w}x{h})")

    # ── Step 1: YOLO ─────────────────────────────────────────────────────
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, size=TRAIN_IMG_SIZE)
    preds   = results.xyxy[0].cpu().numpy()

    print(f"  YOLO  : {len(preds)} card(s) located")

    if len(preds) == 0:
        print("  [RESULT] No bank cards detected.")
        print("  Tip — if a card is visible, try a lower --conf (e.g. 0.05)")
        return 0, 0

    annotated  = img_bgr.copy()
    fake_count = 0
    real_count = 0

    for i, (*xyxy, conf, cls_id) in enumerate(preds):
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        conf   = float(conf)

        print(f"\n  ── Card #{i+1}  conf={conf:.3f}  bbox=[{x1},{y1} → {x2},{y2}]")

        # ── Step 2: OCR ───────────────────────────────────────────────────
        ocr_issues = []
        if use_ocr:
            crop = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size > 0:
                ocr_issues, raw_text = ocr_validate(crop)
                print(f"  OCR text  : {repr(raw_text[:150])}")
            else:
                print("  [WARN] Empty crop — bbox out of bounds")

        is_fake = len(ocr_issues) > 0

        if is_fake:
            fake_count += 1
            print(f"  [RESULT] FAKE CARD — {len(ocr_issues)} issue(s):")
            for issue in ocr_issues:
                print(f"    • [{issue['type']}] {issue['message']}")
                if issue.get("detail"):
                    print(f"      → {issue['detail']}")
        else:
            real_count += 1
            if use_ocr:
                print("  [RESULT] REAL CARD — all checks passed")
            else:
                print("  [RESULT] CARD DETECTED (OCR disabled)")

        # ── Draw bounding box ─────────────────────────────────────────────
        color = (0, 50, 220) if is_fake else (0, 200, 80)
        label = f"{'FAKE' if is_fake else 'OK'}  {conf:.0%}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for j, issue in enumerate(ocr_issues):
            cv2.putText(annotated, issue["type"], (x1, y2 + 20 + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    print(f"\n  Summary : {len(preds)} card(s)  |  FAKE: {fake_count}  |  REAL: {real_count}")

    if save:
        out_path = image_path.parent / f"result_{image_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved   : {out_path}")

    if show:
        cv2.imshow(f"Result — {image_path.name}", annotated)
        print("  [Press any key to continue...]")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return fake_count, real_count


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fake Credential Detector — local test (YOLO + OCR)"
    )
    parser.add_argument("--model",  required=True, help="Path to best.pt")
    parser.add_argument("--image",  default=None,  help="Single image to test")
    parser.add_argument("--folder", default=None,  help="Folder of images to test")
    parser.add_argument("--conf",   type=float, default=0.20,
                        help="YOLO confidence threshold (default 0.20)")
    parser.add_argument("--iou",    type=float, default=0.45,
                        help="NMS IoU threshold (default 0.45)")
    parser.add_argument("--show",   action="store_true",
                        help="Show annotated image in a window")
    parser.add_argument("--save",   action="store_true",
                        help="Save annotated image as result_<name>.jpg")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Skip OCR — only show YOLO card locations")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide --image or --folder")

    print("=" * 60)
    print("  Fake Credential Detector — Local Test")
    print("=" * 60)

    model = load_model(args.model, args.conf, args.iou)

    images = []
    if args.image:
        images.append(Path(args.image))
    if args.folder:
        d = Path(args.folder)
        images += list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))

    if not images:
        print("[ERROR] No images found.")
        sys.exit(1)

    print(f"\n[INFO]  Images   : {len(images)}")
    print(f"[INFO]  OCR      : {'OFF (--no-ocr)' if args.no_ocr else 'ON (pytesseract)'}")

    total_fake = total_real = 0
    for img_path in images:
        f, r = process_image(model, img_path,
                             use_ocr=not args.no_ocr,
                             show=args.show,
                             save=args.save)
        total_fake += f
        total_real += r

    if len(images) > 1:
        print(f"\n{'━'*60}")
        print(f"  TOTAL — FAKE: {total_fake}  |  REAL: {total_real}")

    print(f"{'━'*60}")
    print("  Done.")


if __name__ == "__main__":
    main()