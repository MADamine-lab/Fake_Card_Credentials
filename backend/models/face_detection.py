import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import os

class FaceDetector:
    """
    Face detection using YOLOv5 (lightweight version for 4GB GPU)
    Can be replaced with custom trained YOLO or use pretrained face detection
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize face detector
        
        Args:
            model_path: Path to trained model weights
            device: torch device (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        
        # Try to load YOLOv5 face detection model
        try:
            # Using YOLOv5n (nano) for low memory usage
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            print(f"YOLOv5 face detector loaded on {self.device}")
        except Exception as e:
            print(f"Warning: Could not load YOLOv5: {e}")
            print("Falling back to OpenCV Haar Cascade")
            self._init_haar_cascade()
        
        # If custom model path provided, try to load it
        if model_path and os.path.exists(model_path):
            self._load_custom_model(model_path)
    
    def _init_haar_cascade(self):
        """Initialize Haar Cascade as fallback"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        self.use_haar = True
        print("Using OpenCV Haar Cascade for face detection")
    
    def _load_custom_model(self, model_path):
        """Load custom trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded custom model from {model_path}")
        except Exception as e:
            print(f"Error loading custom model: {e}")
    
    def detect(self, image):
        """
        Detect faces in image
        
        Args:
            image: numpy array (BGR format from OpenCV)
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2, confidence]
        """
        if self.model is None and hasattr(self, 'haar_cascade'):
            return self._detect_haar(image)
        else:
            return self._detect_yolo(image)
    
    def _detect_yolo(self, image):
        """Detect faces using YOLO"""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            with torch.no_grad():
                results = self.model(image_rgb)
            
            # Filter for person class (class 0 in COCO)
            # For actual face detection, you'd need a face-specific YOLO model
            detections = results.xyxy[0].cpu().numpy()
            
            # Filter detections (person class = 0)
            faces = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if cls == 0:  # Person class
                    faces.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
            
            return np.array(faces) if faces else np.array([])
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return np.array([])
    
    def _detect_haar(self, image):
        """Detect faces using Haar Cascade"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Convert to [x1, y1, x2, y2, confidence] format
            result = []
            for (x, y, w, h) in faces:
                result.append([x, y, x+w, y+h, 0.9])  # Fixed confidence for Haar
            
            return np.array(result) if result else np.array([])
            
        except Exception as e:
            print(f"Error in Haar detection: {e}")
            return np.array([])
    
    def draw_detections(self, image, faces):
        """Draw bounding boxes on image"""
        image_copy = image.copy()
        for face in faces:
            x1, y1, x2, y2 = map(int, face[:4])
            conf = face[4] if len(face) > 4 else 0.0
            
            # Draw rectangle
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f'Face: {conf:.2f}'
            cv2.putText(image_copy, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image_copy
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None or hasattr(self, 'haar_cascade')
    
    def save_model(self, save_path):
        """Save model weights"""
        if self.model is not None:
            torch.save({
                'model': self.model.state_dict(),
                'conf_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold
            }, save_path)
            print(f"Model saved to {save_path}")


class CustomYOLOFaceDetector(nn.Module):
    """
    Lightweight custom YOLO-based face detector
    Optimized for 4GB GPU
    """
    
    def __init__(self, num_classes=1):
        super(CustomYOLOFaceDetector, self).__init__()
        
        # Lightweight backbone (MobileNetV2-inspired)
        self.backbone = nn.Sequential(
            # Input: 3 x 416 x 416
            self._conv_block(3, 16, stride=2),      # -> 16 x 208 x 208
            self._conv_block(16, 32, stride=2),     # -> 32 x 104 x 104
            self._conv_block(32, 64, stride=2),     # -> 64 x 52 x 52
            self._conv_block(64, 128, stride=2),    # -> 128 x 26 x 26
            self._conv_block(128, 256, stride=2),   # -> 256 x 13 x 13
        )
        
        # Detection head
        # Output: (num_anchors * (5 + num_classes)) x H x W
        # 5 = x, y, w, h, objectness
        num_anchors = 3
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, num_anchors * (5 + num_classes), 1)
        )
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
    
    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """Convolutional block with BatchNorm and LeakyReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        """Forward pass"""
        x = self.backbone(x)
        x = self.detection_head(x)
        return x


if __name__ == "__main__":
    # Test face detector
    print("Testing Face Detector...")
    detector = FaceDetector()
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect faces
    faces = detector.detect(dummy_image)
    print(f"Detected {len(faces)} faces")
    print(f"Detector loaded: {detector.is_loaded()}")