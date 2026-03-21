import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
import re
from pathlib import Path

class TextClassifier:
    """
    Hybrid text classifier for sensitive message detection
    Uses both RandomForest with TF-IDF and optional neural network
    """
    
    def __init__(self, model_path=None, use_neural=False):
        """
        Initialize text classifier
        
        Args:
            model_path: Path to saved model
            use_neural: Use neural network instead of RandomForest
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        self.use_neural = use_neural
        
        if use_neural:
            self.model = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            # RandomForest with optimized parameters for GPU memory
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'
            )
        
        self.is_trained = False
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Sensitive keywords for rule-based filtering
        self.sensitive_keywords = [
            'password', 'credit card', 'ssn', 'social security',
            'confidential', 'secret', 'private', 'bank account',
            'api key', 'token', 'credential', 'sensitive'
        ]
    
    def preprocess_text(self, text):
        """Preprocess text for classification"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, texts):
        """Extract TF-IDF features from texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess
        processed_texts = [self.preprocess_text(t) for t in texts]
        
        # Vectorize
        if not hasattr(self.vectorizer, 'vocabulary_'):
            # Fit and transform
            features = self.vectorizer.fit_transform(processed_texts)
        else:
            # Only transform
            features = self.vectorizer.transform(processed_texts)
        
        return features
    
    def has_sensitive_keywords(self, text):
        """Check if text contains sensitive keywords"""
        text_lower = text.lower()
        for keyword in self.sensitive_keywords:
            if keyword in text_lower:
                return True, 1.0
        return False, 0.0
    
    def train(self, texts, labels, validation_split=0.2):
        """
        Train the classifier
        
        Args:
            texts: List of text samples
            labels: List of labels (0=normal, 1=sensitive)
            validation_split: Fraction of data for validation
        """
        print("Training text classifier...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        # Extract features
        print("Extracting features...")
        X_train_features = self.extract_features(X_train)
        X_val_features = self.extract_features(X_val)
        
        if self.use_neural:
            return self._train_neural(X_train_features, y_train, X_val_features, y_val)
        else:
            return self._train_random_forest(X_train_features, y_train, X_val_features, y_val)
    
    def _train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train RandomForest model"""
        print("Training RandomForest...")
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        }
    
    def _train_neural(self, X_train, y_train, X_val, y_val):
        """Train neural network model"""
        print("Training Neural Network...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.toarray()).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val.toarray()).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Initialize model
        input_size = X_train.shape[1]
        self.model = SensitiveTextNN(input_size).to(self.device)
        
        # AdamW optimizer with weight decay for regularization
        optimizer = AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.999)
        )
        
        # Training loop
        num_epochs = 50
        batch_size = 32
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            
            # Mini-batch training
            indices = torch.randperm(len(X_train_tensor))
            epoch_loss = 0.0
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X_train_tensor[batch_indices]
                batch_y = y_train_tensor[batch_indices]
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = F.cross_entropy(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (regularization)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = F.cross_entropy(val_outputs, y_val_tensor)
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_acc = (val_predictions == y_val_tensor).float().mean().item()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(indices):.4f}, Val Acc: {val_acc:.4f}")
        
        self.is_trained = True
        
        return {
            'train_accuracy': None,
            'val_accuracy': best_val_acc
        }
    
    def predict(self, text):
        """
        Predict if text is sensitive
        
        Args:
            text: Input text
            
        Returns:
            (is_sensitive, confidence)
        """
        # First check for sensitive keywords
        has_keywords, keyword_conf = self.has_sensitive_keywords(text)
        if has_keywords:
            return True, keyword_conf
        
        if not self.is_trained:
            # Fallback to keyword-based detection
            return has_keywords, keyword_conf
        
        # Extract features
        features = self.extract_features(text)
        
        if self.use_neural:
            return self._predict_neural(features)
        else:
            return self._predict_random_forest(features)
    
    def _predict_random_forest(self, features):
        """Predict using RandomForest"""
        # Get prediction and probability
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        return bool(prediction), float(confidence)
    
    def _predict_neural(self, features):
        """Predict using neural network"""
        self.model.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features.toarray()).to(self.device)
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        return bool(prediction), float(confidence)
    
    def save_model(self, save_path):
        """Save model and vectorizer"""
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_neural:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vectorizer': self.vectorizer,
                'is_trained': self.is_trained
            }, save_path)
        else:
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'is_trained': self.is_trained
                }, f)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """Load model and vectorizer"""
        try:
            if self.use_neural:
                checkpoint = torch.load(model_path, map_location=self.device)
                input_size = checkpoint['vectorizer'].max_features
                self.model = SensitiveTextNN(input_size).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.vectorizer = checkpoint['vectorizer']
                self.is_trained = checkpoint['is_trained']
            else:
                with open(model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.model = checkpoint['model']
                self.vectorizer = checkpoint['vectorizer']
                self.is_trained = checkpoint['is_trained']
            
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def is_loaded(self):
        """Check if model is loaded and ready"""
        return self.is_trained


class SensitiveTextNN(nn.Module):
    """
    Neural network for sensitive text classification
    Optimized for 4GB GPU with regularization techniques
    """
    
    def __init__(self, input_size, hidden_size=256, num_classes=2, dropout=0.5):
        super(SensitiveTextNN, self).__init__()
        
        # Network architecture with dropout for regularization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        
        return x


if __name__ == "__main__":
    # Test text classifier
    print("Testing Text Classifier...")
    
    # Test with RandomForest
    classifier_rf = TextClassifier(use_neural=False)
    
    # Test prediction
    test_texts = [
        "Hello, how are you?",
        "My password is 12345",
        "Please send me the confidential report"
    ]
    
    for text in test_texts:
        is_sensitive, confidence = classifier_rf.predict(text)
        print(f"Text: '{text}'")
        print(f"Sensitive: {is_sensitive}, Confidence: {confidence:.4f}\n")