"""
Multi-Object Detector for Faces, Credit Cards, and ID Cards
Optimized for 4GB GPU
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import os
import re

class MultiObjectDetector:
    """
    Détecte:
    - Visages (faces)
    - Cartes bancaires (credit cards)
    - Cartes d'identité (ID cards)
    """
    
    OBJECT_TYPES = {
        0: 'face',
        1: 'credit_card',
        2: 'id_card'
    }
    
    def __init__(self, model_path=None, device=None, use_simple=True):
        """
        Initialize multi-object detector
        
        Args:
            model_path: Path to trained model weights
            device: torch device (cuda/cpu)
            use_simple: Use simple detectors (recommended for 4GB GPU)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_simple = use_simple
        
        # Initialize detectors
        if use_simple:
            print("Using simple multi-object detector (optimized for 4GB GPU)")
            self._init_simple_detectors()
        else:
            print("Using YOLO-based multi-object detector")
            self._init_yolo_detector()
    
    def _init_simple_detectors(self):
        """Initialize simple detectors using OpenCV and pattern matching"""
        # Face detector (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Card patterns (regex for text detection)
        self.card_patterns = {
            'credit_card': [
                r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',  # 16 digits
                r'\b\d{4}[\s\-]?\d{6}[\s\-]?\d{5}\b',  # 15 digits (Amex)
            ],
            'expiry': r'\b(0[1-9]|1[0-2])[\s\/\-]?(\d{2}|\d{4})\b',  # MM/YY or MM/YYYY
            'cvv': r'\b\d{3,4}\b'
        }
        
        # Card visual features
        self.card_aspect_ratio = (1.586, 0.1)  # Standard card ratio 85.6 x 53.98 mm
        self.min_card_area = 5000  # Minimum pixels for card detection
        
        print("Simple detectors initialized")
    
    def _init_yolo_detector(self):
        """Initialize YOLO detector for all objects"""
        try:
            # Check if custom trained model exists
            custom_model_path = Path(__file__).parent.parent.parent / 'training' / 'runs' / 'multi_object_detector' / 'weights' / 'best.pt'
            
            if custom_model_path.exists():
                print(f"Loading custom trained YOLOv5 model from {custom_model_path}")
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                           path=str(custom_model_path), force_reload=False)
                self.use_custom_yolo = True
            else:
                # Use pretrained YOLOv5
                print("Loading pretrained YOLOv5n model...")
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
                self.use_custom_yolo = False
            
            self.model.to(self.device)
            self.model.eval()
            
            # Set confidence threshold
            self.model.conf = 0.5  # Confidence threshold
            self.model.iou = 0.45   # NMS IOU threshold
            
            print(f"YOLOv5 loaded on {self.device}")
            print(f"Custom model: {self.use_custom_yolo}")
            
        except Exception as e:
            print(f"Could not load YOLO: {e}")
            print("Falling back to simple detectors")
            self.use_simple = True
            self._init_simple_detectors()
    
    def detect(self, image):
        """
        Detect all objects in image
        
        Args:
            image: numpy array (BGR format from OpenCV)
            
        Returns:
            Dict with detections:
            {
                'faces': [[x1, y1, x2, y2, confidence], ...],
                'credit_cards': [[x1, y1, x2, y2, confidence], ...],
                'id_cards': [[x1, y1, x2, y2, confidence], ...]
            }
        """
        if self.use_simple:
            return self._detect_simple(image)
        else:
            return self._detect_yolo(image)
    
    def _detect_simple(self, image):
        """Detect using simple methods"""
        detections = {
            'faces': [],
            'credit_cards': [],
            'id_cards': []
        }
        
        # Detect faces
        faces = self._detect_faces_haar(image)
        detections['faces'] = faces
        
        # Detect cards (by shape and text)
        cards = self._detect_cards_simple(image)
        detections['credit_cards'] = cards['credit_cards']
        detections['id_cards'] = cards['id_cards']
        
        return detections
    
    def _detect_faces_haar(self, image):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        result = []
        for (x, y, w, h) in faces:
            result.append([x, y, x+w, y+h, 0.9])
        
        return result
    
    def _detect_cards_simple(self, image):
        """Detect cards using shape and edge detection"""
        cards = {
            'credit_cards': [],
            'id_cards': []
        }
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Check if it could be a card
            if area < self.min_card_area:
                continue
            
            aspect_ratio = w / h if h > 0 else 0
            
            # Credit card aspect ratio: ~1.586 (85.6mm x 53.98mm)
            # ID card aspect ratio: varies by country
            
            # Check for credit card shape
            if abs(aspect_ratio - 1.586) < 0.2 and area > 10000:
                # Additional check: look for text patterns in region
                roi = image[y:y+h, x:x+w]
                if self._contains_card_patterns(roi):
                    cards['credit_cards'].append([x, y, x+w, y+h, 0.75])
            
            # Check for ID card shape (more rectangular, varies)
            elif 1.4 < aspect_ratio < 1.8 and area > 15000:
                roi = image[y:y+h, x:x+w]
                if self._contains_id_patterns(roi):
                    cards['id_cards'].append([x, y, x+w, y+h, 0.7])
        
        return cards
    
    def _contains_card_patterns(self, image_roi):
        """Check if region contains credit card patterns"""
        try:
            # Try OCR with pytesseract if available
            try:
                import pytesseract
                text = pytesseract.image_to_string(image_roi)
            except:
                # Fallback: simple text detection using template matching
                return self._simple_text_check(image_roi)
            
            # Check for card number pattern
            for pattern in self.card_patterns['credit_card']:
                if re.search(pattern, text):
                    return True
            
            # Check for expiry date
            if re.search(self.card_patterns['expiry'], text):
                return True
            
            # Check for common card keywords
            keywords = ['VISA', 'MASTERCARD', 'AMEX', 'DISCOVER', 'DEBIT', 'CREDIT']
            text_upper = text.upper()
            for keyword in keywords:
                if keyword in text_upper:
                    return True
            
            return False
        except:
            return False
    
    def _contains_id_patterns(self, image_roi):
        """Check if region contains ID card patterns"""
        try:
            # Look for ID-specific patterns
            try:
                import pytesseract
                text = pytesseract.image_to_string(image_roi)
            except:
                return self._simple_text_check(image_roi)
            
            # Common ID card keywords
            id_keywords = [
                'ID', 'IDENTITY', 'PASSPORT', 'LICENSE', 'PERMIT',
                'CARTE', 'IDENTITE', 'PERMIS', 'NATIONAL',
                'DATE OF BIRTH', 'DOB', 'EXPIRES', 'ISSUED'
            ]
            
            text_upper = text.upper()
            for keyword in id_keywords:
                if keyword in text_upper:
                    return True
            
            return False
        except:
            return False
    
    def _simple_text_check(self, image_roi):
        """Simple check for presence of text (no OCR)"""
        # Use edge density as proxy for text
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Text regions typically have higher edge density
        return edge_density > 0.05
    
    def _detect_yolo(self, image):
        """Detect using YOLO (if available)"""
        detections = {
            'faces': [],
            'credit_cards': [],
            'id_cards': []
        }
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            with torch.no_grad():
                results = self.model(image_rgb)
            
            # Parse results
            preds = results.xyxy[0].cpu().numpy()
            
            for pred in preds:
                x1, y1, x2, y2, conf, cls = pred
                cls = int(cls)
                
                if hasattr(self, 'use_custom_yolo') and self.use_custom_yolo:
                    # Custom model with 3 classes: 0=face, 1=credit_card, 2=id_card
                    if cls == 0:
                        detections['faces'].append([int(x1), int(y1), int(x2), int(y2), float(conf)])
                    elif cls == 1:
                        detections['credit_cards'].append([int(x1), int(y1), int(x2), int(y2), float(conf)])
                    elif cls == 2:
                        detections['id_cards'].append([int(x1), int(y1), int(x2), int(y2), float(conf)])
                else:
                    # Pretrained COCO model: class 0 = person (approximate face)
                    if cls == 0:
                        detections['faces'].append([int(x1), int(y1), int(x2), int(y2), float(conf)])
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw all detections on image"""
        image_copy = image.copy()
        
        colors = {
            'faces': (0, 255, 0),        # Green
            'credit_cards': (0, 0, 255),  # Red
            'id_cards': (255, 0, 0)       # Blue
        }
        
        for obj_type, color in colors.items():
            for detection in detections.get(obj_type, []):
                x1, y1, x2, y2 = map(int, detection[:4])
                conf = detection[4] if len(detection) > 4 else 0.0
                
                # Draw rectangle
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f'{obj_type}: {conf:.2f}'
                cv2.putText(image_copy, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image_copy
    
    def has_sensitive_objects(self, detections):
        """Check if any sensitive objects were detected"""
        total_objects = (
            len(detections.get('faces', [])) +
            len(detections.get('credit_cards', [])) +
            len(detections.get('id_cards', []))
        )
        return total_objects > 0
    
    def get_detection_summary(self, detections):
        """Get summary of detections"""
        summary = {
            'total_objects': 0,
            'details': []
        }
        
        for obj_type in ['faces', 'credit_cards', 'id_cards']:
            count = len(detections.get(obj_type, []))
            if count > 0:
                summary['total_objects'] += count
                summary['details'].append({
                    'type': obj_type,
                    'count': count,
                    'message': f'{count} {obj_type.replace("_", " ")} détecté(s)'
                })
        
        return summary
    
    def is_loaded(self):
        """Check if detector is loaded"""
        if self.use_simple:
            return self.face_cascade is not None
        else:
            return hasattr(self, 'model') and self.model is not None


if __name__ == "__main__":
    # Test the detector
    print("Testing Multi-Object Detector...")
    
    detector = MultiObjectDetector(use_simple=True)
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect
    detections = detector.detect(test_image)
    
    print(f"Detections: {detections}")
    print(f"Detector loaded: {detector.is_loaded()}")
    
    summary = detector.get_detection_summary(detections)
    print(f"Summary: {summary}")