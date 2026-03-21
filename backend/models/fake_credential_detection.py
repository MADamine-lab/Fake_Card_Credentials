"""
backend/models/fake_credential_detection.py

Architecture: 1-class YOLO + OCR rule engine
=============================================

The trained YOLOv5 model has ONE class only:
    0 = bank_card   (just locates the card — no real/fake classification)

Real vs fake detection is done entirely by the OCR rule engine:
    • Luhn algorithm on the PAN
    • Card-scheme vs BIN-prefix consistency
    • Expiry date sanity check
    • CVV visible on front

A card is flagged FAKE if OCR finds at least one rule violation.
A card is flagged REAL if OCR passes all checks.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

log = logging.getLogger("secureguard.fake_cred")

# ---------------------------------------------------------------------------
# Class map — 1 class, matches training (CLASS_NAMES = ["bank_card"])
# ---------------------------------------------------------------------------
CLASS_NAMES = {0: "bank_card"}

TRAIN_IMG_SIZE = 320     # matches --img-size 320 used during training
DEFAULT_CONF   = 0.20    # lower threshold — real-world images score lower than synthetic
DEFAULT_IOU    = 0.45

_SEARCH_PATHS = [
    Path("training/training/runs/fake_credential_detector/weights/best.pt"),
    Path("training/runs/fake_credential_detector/weights/best.pt"),
    Path("models/trained_models/fake_credential_detector/weights/best.pt"),
    Path(__file__).parent / "trained_models" / "fake_credential_detector" / "weights" / "best.pt",
]

_SCHEME_PREFIXES = {
    "VISA":       ["4"],
    "MASTERCARD": ["51", "52", "53", "54", "55", "2"],
    "AMEX":       ["34", "37"],
}


# ---------------------------------------------------------------------------
# Luhn algorithm
# ---------------------------------------------------------------------------

def luhn_check(pan: str) -> bool:
    """
    Validate a card PAN with the Luhn algorithm.
    Returns True  -> valid (could be real).
    Returns False -> invalid (fake or misread by OCR).
    """
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


# ---------------------------------------------------------------------------
# OCR rule engine — the sole fake/real decision maker
# ---------------------------------------------------------------------------

def validate_card_region(crop_bgr: np.ndarray) -> list:
    """
    Run pytesseract OCR on a cropped card image and apply 4 rules.
    Returns a list of issue dicts — empty list means card looks legitimate.
    """
    issues = []

    try:
        import pytesseract

        # Upscale small crops — Tesseract needs ~300 DPI to read card text
        h, w = crop_bgr.shape[:2]
        scale = max(1.0, 300 / min(h, w))
        if scale > 1.0:
            crop_bgr = cv2.resize(
                crop_bgr, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        thr  = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8,
        )
        text = pytesseract.image_to_string(thr, config="--psm 6 --oem 3")
        log.debug("OCR raw text: %r", text)

        # ── Rule 1: PAN — Luhn check ──────────────────────────────────────
        pan_matches = re.findall(
            r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,4})\b", text
        )
        if pan_matches:
            pan = re.sub(r"\D", "", pan_matches[0])
            if not luhn_check(pan):
                issues.append({
                    "type":    "INVALID_PAN",
                    "message": f"PAN '{pan_matches[0].strip()}' echoue l'algorithme de Luhn",
                })

            # ── Rule 2: Scheme / BIN prefix consistency ───────────────────
            text_upper = text.upper()
            for scheme, prefixes in _SCHEME_PREFIXES.items():
                if scheme in text_upper:
                    if not any(pan.startswith(p) for p in prefixes):
                        issues.append({
                            "type":    "SCHEME_MISMATCH",
                            "message": (
                                f"Logo {scheme} detecte mais le numero "
                                f"'{pan[:6]}...' est incompatible avec ce reseau"
                            ),
                        })
                    break
        else:
            # No PAN readable — suspicious for a card image
            issues.append({
                "type":    "NO_PAN_FOUND",
                "message": "Aucun numero de carte lisible trouve sur l'image",
            })

        # ── Rule 3: Expiry date ───────────────────────────────────────────
        expiry_matches = re.findall(r"\b(\d{1,2})[\/\-](\d{2,4})\b", text)
        if expiry_matches:
            month_str, year_str = expiry_matches[0]
            month = int(month_str)
            year  = int(year_str)
            if year < 100:
                year += 2000
            now = datetime.now()

            if not (1 <= month <= 12):
                issues.append({
                    "type":    "INVALID_EXPIRY_MONTH",
                    "message": f"Mois d'expiration impossible : {month:02d}",
                })
            elif year < now.year or (year == now.year and month < now.month):
                issues.append({
                    "type":    "EXPIRED_CARD",
                    "message": f"Carte expiree depuis {month:02d}/{year}",
                })
            elif year > now.year + 10:
                issues.append({
                    "type":    "INVALID_EXPIRY_YEAR",
                    "message": f"Date d'expiration suspecte : {month:02d}/{year}",
                })

        # ── Rule 4: CVV on front ──────────────────────────────────────────
        if re.search(r"\bCVV\b[\s:]*\d{3,4}", text, re.IGNORECASE):
            issues.append({
                "type":    "CVV_ON_FRONT",
                "message": "CVV imprime au recto — impossible sur une vraie carte",
            })

    except ImportError:
        log.warning(
            "pytesseract not installed — OCR skipped. "
            "pip install pytesseract  (+ install Tesseract binary)"
        )
    except Exception as exc:
        log.error("OCR validation error: %s", exc, exc_info=True)

    return issues


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class FakeCredentialDetector:
    """
    1-class YOLO (locates cards) + OCR rule engine (classifies real/fake).

    Parameters
    ----------
    model_path     : path to best.pt (None = auto-discover)
    conf_threshold : YOLO confidence threshold (default 0.20)
    iou_threshold  : NMS IoU threshold (default 0.45)
    device         : 'cuda' | 'cpu' | None (auto)
    use_ocr        : run OCR checks (default True; requires pytesseract)
    """

    def __init__(
        self,
        model_path     = None,
        conf_threshold : float = DEFAULT_CONF,
        iou_threshold  : float = DEFAULT_IOU,
        device         : str   = None,
        use_ocr        : bool  = True,
    ):
        self._loaded        = False
        self._model         = None
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ocr        = use_ocr

        resolved = self._resolve_path(model_path)
        if resolved is None:
            log.warning(
                "FakeCredentialDetector: no model file found — stub mode. "
                "Train first: python training/train_fake_credential_detector.py --synthetic"
            )
            return
        self._load(resolved)

    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, image_bgr: np.ndarray) -> dict:
        """
        Step 1 — YOLO locates all bank_card regions (class 0).
        Step 2 — OCR rule engine checks each crop for fake credentials.

        Returns
        -------
        {
          cards_found    : int   total cards located by YOLO
          fake_cards     : int   cards that failed >= 1 OCR rule
          real_cards     : int   cards that passed all OCR rules
          max_confidence : float highest YOLO confidence score
          detections     : list  one dict per card (see _parse)
        }
        """
        if not self._loaded or self._model is None:
            return self._empty()
        if image_bgr is None or image_bgr.size == 0:
            log.warning("detect() received an empty image.")
            return self._empty()
        try:
            # BGR → RGB: YOLOv5 expects RGB input
            rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self._model(rgb, size=TRAIN_IMG_SIZE)
            return self._parse(results, image_bgr)
        except Exception as exc:
            log.error("detect() failed: %s", exc, exc_info=True)
            return self._empty()

    def detect_and_draw(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run detect() and draw colour-coded boxes. Green = OK, Red = FAKE."""
        result = self.detect(image_bgr)
        out    = image_bgr.copy()
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            is_fake  = det["is_fake"]
            conf     = det["confidence"]
            issues   = det.get("ocr_issues", [])
            color    = (0, 50, 220) if is_fake else (0, 200, 80)
            label    = f"{'FAKE' if is_fake else 'OK'} {conf:.0%}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            for j, issue in enumerate(issues):
                cv2.putText(out, issue["type"], (x1, y2 + 16 + j * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return out

    # ── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_path(explicit):
        if explicit is not None:
            p = Path(explicit)
            if p.exists():
                return p
            log.warning("Explicit model path not found: %s", p.absolute())
        for candidate in _SEARCH_PATHS:
            if candidate.exists():
                log.info("Model found: %s", candidate)
                return candidate
        return None

    def _load(self, path: Path) -> None:
        try:
            log.info("Loading FakeCredentialDetector from %s ...", path)
            yolov5_dir = Path("training/yolov5")
            if not yolov5_dir.exists():
                yolov5_dir = Path(__file__).parent.parent.parent / "training" / "yolov5"

            if yolov5_dir.exists():
                self._model = torch.hub.load(
                    str(yolov5_dir), "custom",
                    path=str(path), source="local",
                    force_reload=False, verbose=False,
                )
            else:
                log.warning("Local YOLOv5 not found — falling back to torch.hub (needs internet)")
                self._model = torch.hub.load(
                    "ultralytics/yolov5", "custom",
                    path=str(path), force_reload=False, verbose=False,
                )

            self._model.to(self.device)
            self._model.conf = self.conf_threshold
            self._model.iou  = self.iou_threshold
            self._model.eval()

            log.info(
                "Model ready — device=%s  conf=%.2f  img_size=%d  classes=%s  ocr=%s",
                self.device, self.conf_threshold, TRAIN_IMG_SIZE,
                getattr(self._model, "names", "?"), self.use_ocr,
            )
            self._loaded = True

        except Exception as exc:
            log.error("Failed to load model: %s", exc, exc_info=True)
            self._loaded = False

    def _parse(self, results, original_bgr: np.ndarray) -> dict:
        """
        Convert raw YOLO output into structured result.
        OCR runs on each detected card crop to decide real vs fake.
        """
        detections = []
        fake_count = 0
        max_conf   = 0.0

        raw = results.xyxy[0].cpu().numpy() if len(results.xyxy) else np.zeros((0, 6))

        for *xyxy, conf, cls_id in raw:
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            conf    = float(conf)
            cls_id  = int(cls_id)
            max_conf = max(max_conf, conf)

            # Stage 2: OCR rule engine on the card crop
            ocr_issues = []
            if self.use_ocr:
                h, w = original_bgr.shape[:2]
                crop = original_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size > 0:
                    ocr_issues = validate_card_region(crop)

            is_fake = len(ocr_issues) > 0
            if is_fake:
                fake_count += 1

            detections.append({
                "class_id":   cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "confidence": round(conf, 4),
                "bbox":       [x1, y1, x2, y2],
                "is_fake":    is_fake,
                "ocr_issues": ocr_issues,
                "issues":     ocr_issues,   # kept for app.py compatibility
            })

        cards_found = len(detections)
        real_count  = cards_found - fake_count

        log.info(
            "Result — cards=%d  fake=%d  real=%d  max_conf=%.2f",
            cards_found, fake_count, real_count, max_conf,
        )
        return {
            "cards_found":    cards_found,
            "fake_cards":     fake_count,
            "real_cards":     real_count,
            "max_confidence": round(max_conf, 4),
            "detections":     detections,
        }

    @staticmethod
    def _empty() -> dict:
        return {
            "cards_found": 0, "fake_cards": 0,
            "real_cards": 0, "max_confidence": 0.0, "detections": [],
        }