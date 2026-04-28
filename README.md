# Fake Card Credential Detection System

Bank card forged credential detection using a two-stage pipeline:
- Stage 1: YOLOv5 object detection for card region localization
- Stage 2: OCR + rule-based validation (PAN Luhn, BIN-to-scheme, expiry sanity, CVV-on-front detection)

## Project Overview

This repository contains a complete proof-of-concept for detecting fake credit card credentials in images. It includes dataset preparation, model training, inference pipeline, and a Flask REST API with a simple frontend.

Key features:
- Synthetic dataset generation for realistic fraud scenarios
- YOLOv5n object detector trained on a 400+ image synthetic dataset
- OCR pipeline using `pytesseract` and OpenCV
- Multi-rule credential validator for PAN, issuer consistency, expiry date, and CVV placement
- Flask API with optional OTP lockout on fraud detection
- Frontend interface in `frontend/` for demo use

## Repo Structure

- `backend/`
  - `app.py` - Flask API service
  - `config.py` - app settings
  - `utils.py` - helper functions
  - `models/` - detection and classification pipeline code
- `training/`
  - `prepare_datasets.py` - synthetic dataset setup
  - `test_detector.py` - inference test script
  - `train_fake_credential_detector.py` - training script
  - `yolov5/` - YOLOv5 model code and configs
- `frontend/`
  - `index.html`, `script.js`, `style.css`

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but strongly recommended for training)
- Tesseract OCR (installed and accessible in PATH)

### Python dependencies

```
cd backend
python -m pip install -r requirements.txt
```

Training requirements may be in `training/yolov5/requirements.txt`.

## Quick Start (Inference)

Example command:

```bash
cd training
python test_detector.py --model "training/runs/fake_credential_detector/weights/best.pt" --image visa_card.jpg --show
```

This runs detection on `fake_card.jpg` + OCR validation and displays the result.

## Training

1. Prepare datasets:
   - run `python prepare_datasets.py` (adjust dataset paths as needed)

2. Train YOLOv5 model:
   - `python train_fake_credential_detector.py`
   - or use YOLOv5's standard `training/yolov5/train.py` with custom config

3. Model outputs saved under `training/training/runs/fake_credential_detector/`.

## API

Start Flask backend:

```bash
cd backend
python app.py
```

Then use the REST endpoint (likely on `http://localhost:5000`) to upload images and get detection/validation results.

## Frontend

Open `frontend/index.html` in a browser. It should call the backend endpoint via `script.js` for demo detection.

## Evaluation Metrics

Reported performance on synthetic validation:
- Precision: 99.1%
- Recall: 98.2%
- mAP@50: 99.2%

## Notes

- The dataset and model are experimental and intended for research/demo only.
- Do not use in production without thorough ethical, privacy, and security reviews.

## Contact

For questions, refer to repo owner or maintainer.
