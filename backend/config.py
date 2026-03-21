import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model paths
MODELS_DIR = BASE_DIR / 'models' / 'trained_models'
FACE_MODEL_PATH = MODELS_DIR / 'face_detector.pth'
TEXT_MODEL_PATH = MODELS_DIR / 'text_classifier.pkl'

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Application settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
DEBUG = os.environ.get('DEBUG', 'True') == 'True'

# Security settings
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
AUTH_CODE_LENGTH = 6
AUTH_CODE_EXPIRY_MINUTES = 5
MAX_AUTH_ATTEMPTS = 3

# Model settings
FACE_DETECTION_CONFIDENCE = 0.5
TEXT_CLASSIFICATION_THRESHOLD = 0.7
USE_NEURAL_TEXT_CLASSIFIER = False  # Set to True to use neural network instead of RandomForest

# Face Detection Settings
# For 4GB GPU, options are:
# - 'haar': OpenCV Haar Cascade (fastest, no GPU needed, 90% accuracy)
# - 'yolo': YOLOv5 nano (slower, uses GPU, 95% accuracy)
# - 'custom': Your trained model
FACE_DETECTOR_TYPE = 'haar'  # Recommended for 4GB GPU

# Rate limiting
RATE_LIMIT_ENABLED = True
MAX_REQUESTS_PER_MINUTE = 10

# Email configuration for sending authentication codes
EMAIL_CONFIG = {
    'enabled': True,  # Set to True to send emails
    'recipient': 'mohamedaminedardouri1@gmail.com',  # Your email
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': os.environ.get('SENDER_EMAIL', 'noreply@securedata.com'),
    'sender_password': os.environ.get('EMAIL_PASSWORD', ''),  # Set via environment variable
    'use_tls': True
}

# Dataset paths for training
DATASET_DIR = BASE_DIR.parent / 'training' / 'datasets'
FACE_DATASET_PATH = DATASET_DIR / 'wider_face'
TEXT_DATASET_PATH = DATASET_DIR / 'toxic_comments'

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.01  # L2 regularization
DROPOUT_RATE = 0.5
EARLY_STOPPING_PATIENCE = 10

# Device settings
DEVICE = 'cuda'  # or 'cpu'

# Logging
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FILE = BASE_DIR / 'app.log'

def get_config():
    """Get configuration dictionary"""
    return {
        'base_dir': BASE_DIR,
        'models_dir': MODELS_DIR,
        'face_model_path': FACE_MODEL_PATH,
        'text_model_path': TEXT_MODEL_PATH,
        'secret_key': SECRET_KEY,
        'debug': DEBUG,
        'max_content_length': MAX_CONTENT_LENGTH,
        'allowed_extensions': ALLOWED_EXTENSIONS,
        'auth_code_length': AUTH_CODE_LENGTH,
        'auth_code_expiry_minutes': AUTH_CODE_EXPIRY_MINUTES,
        'max_auth_attempts': MAX_AUTH_ATTEMPTS,
        'face_detection_confidence': FACE_DETECTION_CONFIDENCE,
        'text_classification_threshold': TEXT_CLASSIFICATION_THRESHOLD,
        'use_neural_text_classifier': USE_NEURAL_TEXT_CLASSIFIER,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'weight_decay': WEIGHT_DECAY,
        'dropout_rate': DROPOUT_RATE,
        'device': DEVICE,
    }

if __name__ == "__main__":
    config = get_config()
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")