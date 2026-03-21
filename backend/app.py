"""
SecureGuard — Bank Card Fake Credential Detection API
Flask backend focused exclusively on detecting real vs fake bank cards
using the trained YOLOv5 FakeCredentialDetector.

Routes
------
GET  /api/health         — liveness + model status
POST /api/send-message   — analyse image for fake cards, gate on OTP if found
POST /api/verify-code    — validate OTP and release pending message
POST /api/analyze        — dry-run analysis (no OTP, no storage)
GET  /api/stats          — server statistics
GET  /                   — serve frontend
"""

import io
import os
import base64
import logging
import secrets
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("secureguard")

# ---------------------------------------------------------------------------
# Model import — fake credential / bank-card detector only
# ---------------------------------------------------------------------------
try:
    from models.fake_credential_detection import FakeCredentialDetector
except ImportError as e:
    log.warning("FakeCredentialDetector unavailable: %s", e)
    FakeCredentialDetector = None

from utils import generate_auth_code, send_auth_code_email

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    from config import EMAIL_CONFIG
except Exception:
    EMAIL_CONFIG = {
        "enabled":   False,
        "recipient": "mohamedaminedardouri1@gmail.com",
    }

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"]        = 16 * 1024 * 1024   # 16 MB
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=24)

CORS(app, supports_credentials=True)

# ---------------------------------------------------------------------------
# In-memory stores (swap for Redis / DB in production)
# ---------------------------------------------------------------------------
auth_codes:       dict = {}   # message_id -> {code, expires, user_id}
pending_messages: dict = {}   # message_id -> {image, user_id, timestamp}

# ---------------------------------------------------------------------------
# Model initialisation — fake credential detector only
# ---------------------------------------------------------------------------
log.info("Initialising FakeCredentialDetector...")

fake_cred_detector = None
if FakeCredentialDetector is not None:
    try:
        model_path = Path("models/trained_models/fake_credential_detector/weights/best.pt")
        fake_cred_detector = FakeCredentialDetector(
            model_path=str(model_path) if model_path.exists() else None
        )
        log.info("FakeCredentialDetector ready (model found: %s).", model_path.exists())
    except Exception as exc:
        log.error("FakeCredentialDetector init failed: %s", exc)

log.info("Model initialisation complete.")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def decode_image(image_data: str) -> np.ndarray:
    """
    Decode a base64 image string (with or without data-URI prefix)
    into a BGR numpy array ready for OpenCV / YOLOv5.
    Raises ValueError on failure.
    """
    try:
        raw       = image_data.split(",")[1] if "," in image_data else image_data
        img_bytes = base64.b64decode(raw)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr       = np.array(pil_image)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc


def purge_expired() -> None:
    """Remove expired OTP entries to prevent memory leaks."""
    now   = datetime.now()
    stale = [mid for mid, v in auth_codes.items() if now > v["expires"]]
    for mid in stale:
        auth_codes.pop(mid, None)
        pending_messages.pop(mid, None)
    if stale:
        log.debug("Purged %d expired OTP entries.", len(stale))


def create_pending(image_data, user_id: str):
    """Store a pending message + OTP. Returns (message_id, auth_code)."""
    purge_expired()
    auth_code  = generate_auth_code()
    message_id = secrets.token_urlsafe(16)
    auth_codes[message_id] = {
        "code":    auth_code,
        "expires": datetime.now() + timedelta(minutes=5),
        "user_id": user_id,
    }
    pending_messages[message_id] = {
        "image":     image_data,
        "user_id":   user_id,
        "timestamp": datetime.now().isoformat(),
    }
    return message_id, auth_code


def try_send_email(auth_code: str):
    """Attempt to deliver OTP via email. Returns (success, message)."""
    if not EMAIL_CONFIG.get("enabled", False):
        return False, "Email disabled (development mode)"
    return send_auth_code_email(
        code=auth_code,
        recipient_email=EMAIL_CONFIG["recipient"],
        smtp_config=EMAIL_CONFIG,
    )


def build_pending_response(message_id: str, auth_code: str, detected_issues: list) -> dict:
    """
    Build the JSON payload returned when a message is gated by OTP.
    In production (email enabled) the code is hidden; in dev mode it is exposed.
    """
    email_sent, email_msg = try_send_email(auth_code)

    payload = {
        "status":          "pending",
        "requires_auth":   True,
        "message_id":      message_id,
        "detected_issues": detected_issues,
        "message":         "Code d'authentification requis",
    }

    if EMAIL_CONFIG.get("enabled", False):
        payload["email_sent"]    = email_sent
        payload["email_message"] = email_msg
        if email_sent:
            payload["message"] = "Code envoyé à {}".format(EMAIL_CONFIG["recipient"])
            log.info("OTP emailed to %s", EMAIL_CONFIG["recipient"])
        else:
            payload["auth_code"] = auth_code
            payload["message"]   = "Échec email : {}".format(email_msg)
            log.warning("Email failed (%s) — exposing OTP in response.", email_msg)
    else:
        payload["auth_code"] = auth_code
        payload["message"]   = "Code d'authentification (mode développement)"

    return payload


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check():
    """Liveness probe + model status."""
    return jsonify({
        "status":    "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "fake_cred_detector": fake_cred_detector.is_loaded() if fake_cred_detector else False,
        },
        "pending_messages": len(pending_messages),
    })


@app.route("/api/send-message", methods=["POST"])
def send_message():
    """
    Analyse an image for fake bank cards.

    Flow
    ----
    1. Decode image from base64.
    2. Run FakeCredentialDetector — flags if any fake card is found.
    3. No flags → return success immediately.
    4. Flagged  → store pending msg, generate OTP, optionally email it.
    """
    try:
        data       = request.get_json(force=True)
        image_data = data.get("image")
        user_id    = session.get("user_id", "anonymous")

        if not image_data:
            return jsonify({"error": "Une image est requise pour l'analyse."}), 400

        requires_auth   = False
        detected_issues = []

        # ---- Image decoding -------------------------------------------------
        try:
            image_bgr = decode_image(image_data)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        # ---- Fake credential detection --------------------------------------
        if fake_cred_detector:
            try:
                cred = fake_cred_detector.detect(image_bgr)
                log.info(
                    "FakeCredDetector — real=%d fake=%d",
                    cred.get("real_cards", 0),
                    cred.get("fake_cards", 0),
                )

                if cred.get("real_cards", 0) > 0:
                    # Real cards found — flag for review too (privacy)
                    requires_auth = True
                    detected_issues.append({
                        "type":       "CARTE_REELLE",
                        "count":      cred["real_cards"],
                        "confidence": cred.get("max_confidence", 0.0),
                        "message":    "{} carte(s) bancaire(s) légitime(s) détectée(s) — accès restreint".format(
                            cred["real_cards"]
                        ),
                    })

                if cred.get("fake_cards", 0) > 0:
                    requires_auth = True
                    detected_issues.append({
                        "type":       "CARTE_FALSIFIEE",
                        "count":      cred["fake_cards"],
                        "confidence": cred.get("max_confidence", 0.0),
                        "message":    "{} carte(s) avec credentials falsifiés détectée(s)".format(
                            cred["fake_cards"]
                        ),
                    })

            except Exception as exc:
                log.error("FakeCredentialDetector error: %s", exc)
                return jsonify({"error": "Erreur lors de l'analyse de l'image."}), 500
        else:
            log.warning("FakeCredentialDetector not loaded — cannot analyse image.")
            return jsonify({"error": "Modèle de détection non disponible."}), 503

        # ---- Decision -------------------------------------------------------
        if not requires_auth:
            log.info("Image cleared — no cards detected.")
            return jsonify({
                "status":  "success",
                "message": "Aucune carte bancaire détectée. Image sécurisée.",
            })

        message_id, auth_code = create_pending(image_data, user_id)
        log.info("Image flagged — id=%s issues=%d", message_id, len(detected_issues))
        return jsonify(build_pending_response(message_id, auth_code, detected_issues))

    except Exception as exc:
        log.exception("Unhandled error in /api/send-message: %s", exc)
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/api/verify-code", methods=["POST"])
def verify_code():
    """
    Validate the OTP submitted by the user.

    Checks
    ------
    * message_id exists in auth_codes
    * OTP has not expired (5-minute window)
    * Submitted code matches stored code (constant-time comparison)
    """
    try:
        data       = request.get_json(force=True)
        message_id = (data.get("message_id") or "").strip()
        user_code  = (data.get("code") or "").strip()

        if not message_id or not user_code:
            return jsonify({"error": "message_id et code sont requis"}), 400

        auth_entry = auth_codes.get(message_id)
        if auth_entry is None:
            log.warning("OTP lookup failed — unknown message_id: %s", message_id)
            return jsonify({"error": "Code invalide ou expiré"}), 400

        if datetime.now() > auth_entry["expires"]:
            auth_codes.pop(message_id, None)
            pending_messages.pop(message_id, None)
            log.info("OTP expired for message_id=%s", message_id)
            return jsonify({"error": "Code expiré. Veuillez soumettre une nouvelle image."}), 400

        if not secrets.compare_digest(user_code, auth_entry["code"]):
            log.warning("Incorrect OTP for message_id=%s", message_id)
            return jsonify({"error": "Code incorrect"}), 401

        msg_data = pending_messages.pop(message_id, None)
        auth_codes.pop(message_id, None)

        if msg_data is None:
            log.error("No pending message found for message_id=%s", message_id)
            return jsonify({"error": "Session introuvable"}), 404

        log.info("OTP verified — message_id=%s released.", message_id)
        return jsonify({
            "status":  "success",
            "message": "Accès autorisé — image analysée avec succès",
            "data": {
                "timestamp": msg_data.get("timestamp"),
            },
        })

    except Exception as exc:
        log.exception("Unhandled error in /api/verify-code: %s", exc)
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_content():
    """
    Dry-run analysis — returns detection results without storing
    anything or triggering authentication. Useful for testing.
    """
    try:
        data       = request.get_json(force=True)
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "Une image est requise."}), 400

        try:
            image_bgr = decode_image(image_data)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        results = {"credential_detection": None}

        if fake_cred_detector:
            try:
                cred = fake_cred_detector.detect(image_bgr)
                results["credential_detection"] = {
                    "real_cards":     cred.get("real_cards", 0),
                    "fake_cards":     cred.get("fake_cards", 0),
                    "max_confidence": cred.get("max_confidence", 0.0),
                    "detections":     cred.get("detections", []),
                }
            except Exception as exc:
                results["credential_detection"] = {"error": str(exc)}
        else:
            results["credential_detection"] = {"error": "Modèle non chargé"}

        return jsonify(results)

    except Exception as exc:
        log.exception("Unhandled error in /api/analyze: %s", exc)
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Server statistics used by the frontend."""
    return jsonify({
        "pending_messages": len(pending_messages),
        "active_otps":      len(auth_codes),
        "models_loaded": {
            "fake_cred_detector": fake_cred_detector.is_loaded() if fake_cred_detector else False,
        },
    })


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve the static frontend. Falls back to index.html for SPA routing."""
    frontend_dir = Path(__file__).parent.parent / "frontend"
    target = frontend_dir / path
    if path and target.exists():
        return send_from_directory(str(frontend_dir), path)
    return send_from_directory(str(frontend_dir), "index.html")


# ---------------------------------------------------------------------------
# Global error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def request_too_large(e):
    return jsonify({"error": "Fichier trop volumineux (max 16 Mo)"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route introuvable"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Erreur interne du serveur"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("models/trained_models", exist_ok=True)

    log.info("=" * 60)
    log.info("  SecureGuard — Fake Credential Detector  —  http://0.0.0.0:5000")
    log.info("=" * 60)
    log.info("  FakeCredDetector : %s", fake_cred_detector.is_loaded() if fake_cred_detector else "NOT LOADED")
    log.info("  Email            : %s", "ENABLED" if EMAIL_CONFIG.get("enabled") else "DISABLED (dev mode)")
    log.info("=" * 60)

    app.run(debug=False, host="0.0.0.0", port=5000)