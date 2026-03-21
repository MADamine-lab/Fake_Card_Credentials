"""
Script to download and prepare datasets for training

Datasets recommandés pour GPU 4GB:
1. Face Detection: WIDER FACE (version mini) ou FDDB
2. Text Classification: Jigsaw Toxic Comment ou SMS Spam Collection
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

DATASET_DIR = Path(__file__).parent / 'datasets'
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, destination):
    """Download file from URL"""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded to {destination}")

def extract_zip(zip_path, extract_to):
    """Extract ZIP file"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def prepare_face_dataset():
    """
    Prepare face detection dataset
    
    Option 1: WIDER FACE (subset)
    Kaggle: https://www.kaggle.com/datasets/greatgamedota/widerface
    
    Option 2: FDDB (Face Detection Data Set and Benchmark)
    Plus petit, mieux pour 4GB GPU
    """
    print("\n=== Preparing Face Detection Dataset ===")
    
    face_dir = DATASET_DIR / 'faces'
    face_dir.mkdir(exist_ok=True)
    
    print("""
    Pour télécharger le dataset de détection de visages:
    
    OPTION 1 - WIDER FACE (Recommandé mais plus grand):
    1. Allez sur: https://www.kaggle.com/datasets/greatgamedota/widerface
    2. Téléchargez 'WIDER_train.zip' (version mini si disponible)
    3. Placez le fichier dans: {0}
    
    OPTION 2 - FDDB (Plus petit, 4GB GPU friendly):
    1. Visitez: http://vis-www.cs.umass.edu/fddb/
    2. Téléchargez les images et annotations
    3. Placez dans: {0}
    
    OPTION 3 - Dataset personnalisé:
    Créez un dossier avec:
    - images/ (vos images avec visages)
    - labels/ (fichiers txt avec coordonnées des visages)
    Format label: x1 y1 x2 y2 (un visage par ligne)
    
    Alternative: Utilisez le détecteur Haar Cascade pré-entraîné (pas besoin de dataset)
    """.format(face_dir))
    
    # Create sample structure
    (face_dir / 'images').mkdir(exist_ok=True)
    (face_dir / 'labels').mkdir(exist_ok=True)
    
    print(f"Face dataset directory created: {face_dir}")
    
    return face_dir

def prepare_text_dataset():
    """
    Prepare text classification dataset for sensitive message detection
    
    Datasets recommandés (Kaggle):
    1. Jigsaw Toxic Comment Classification
    2. SMS Spam Collection
    3. Twitter Sentiment Analysis
    """
    print("\n=== Preparing Text Classification Dataset ===")
    
    text_dir = DATASET_DIR / 'text'
    text_dir.mkdir(exist_ok=True)
    
    # Check if we can download SMS Spam Collection (public domain)
    sms_spam_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    try:
        print("Attempting to download SMS Spam Collection dataset...")
        zip_path = text_dir / 'smsspam.zip'
        
        if not zip_path.exists():
            download_file(sms_spam_url, zip_path)
            extract_zip(zip_path, text_dir)
        
        # Load and prepare dataset
        spam_file = text_dir / 'SMSSpamCollection'
        if spam_file.exists():
            df = pd.read_csv(spam_file, sep='\t', names=['label', 'message'])
            
            # Add synthetic sensitive messages
            sensitive_messages = [
                "My password is abc123",
                "Credit card number: 1234-5678-9012-3456",
                "SSN: 123-45-6789",
                "Bank account: 987654321",
                "Confidential report attached",
                "API key: sk_test_123456789",
                "Here is my private information",
                "Secret document for your eyes only"
            ]
            
            # Create balanced dataset
            sensitive_df = pd.DataFrame({
                'label': ['sensitive'] * len(sensitive_messages),
                'message': sensitive_messages
            })
            
            # Convert spam to sensitive label
            df['label'] = df['label'].map({'ham': 'normal', 'spam': 'sensitive'})
            
            # Combine
            combined_df = pd.concat([df, sensitive_df], ignore_index=True)
            
            # Save prepared dataset
            output_file = text_dir / 'prepared_text_dataset.csv'
            combined_df.to_csv(output_file, index=False)
            
            print(f"\nDataset prepared and saved to: {output_file}")
            print(f"Total samples: {len(combined_df)}")
            print(f"Normal: {len(combined_df[combined_df['label'] == 'normal'])}")
            print(f"Sensitive: {len(combined_df[combined_df['label'] == 'sensitive'])}")
            
            return output_file
    
    except Exception as e:
        print(f"Could not automatically download dataset: {e}")
    
    print("""
    Pour un meilleur dataset de classification de texte:
    
    OPTION 1 - Jigsaw Toxic Comment (Kaggle):
    1. Allez sur: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    2. Téléchargez 'train.csv'
    3. Placez dans: {0}
    
    OPTION 2 - SMS Spam Collection (Auto-téléchargé ci-dessus)
    
    OPTION 3 - Dataset personnalisé:
    Créez un fichier CSV avec colonnes: 'message', 'label'
    label: 'normal' ou 'sensitive'
    Sauvegardez comme: {0}/custom_dataset.csv
    """.format(text_dir))
    
    # Create sample dataset
    sample_data = {
        'message': [
            'Hello, how are you?',
            'Meeting at 3pm tomorrow',
            'My password is secret123',
            'Credit card: 1234-5678-9012-3456',
            'Nice weather today',
            'Confidential: Q4 earnings report',
            'What time is the party?',
            'SSN: 123-45-6789'
        ],
        'label': ['normal', 'normal', 'sensitive', 'sensitive', 
                  'normal', 'sensitive', 'normal', 'sensitive']
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_file = text_dir / 'sample_dataset.csv'
    sample_df.to_csv(sample_file, index=False)
    
    print(f"\nSample dataset created: {sample_file}")
    print("You can use this for initial testing")
    
    return sample_file

def create_synthetic_dataset():
    """Create a synthetic dataset for quick testing"""
    print("\n=== Creating Synthetic Dataset ===")
    
    # Synthetic text data with more samples
    normal_messages = [
        "Hello, how are you doing today?",
        "The meeting is scheduled for 3pm",
        "Did you see the game last night?",
        "Let's grab coffee sometime",
        "Happy birthday! Hope you have a great day",
        "The weather is beautiful today",
        "Thanks for your help yesterday",
        "See you at the conference",
        "Great presentation this morning",
        "Looking forward to the weekend"
    ] * 20  # Repeat to get more samples
    
    sensitive_messages = [
        "My password is abc123",
        "Credit card: 1234-5678-9012-3456",
        "SSN: 123-45-6789",
        "Bank account number: 987654321",
        "API key: sk_live_123456789",
        "Confidential financial report",
        "Private medical records attached",
        "Secret project documentation",
        "My PIN code is 4567",
        "Authentication token: Bearer xyz123"
    ] * 20  # Repeat to get more samples
    
    # Create balanced dataset
    all_messages = normal_messages + sensitive_messages
    all_labels = ['normal'] * len(normal_messages) + ['sensitive'] * len(sensitive_messages)
    
    df = pd.DataFrame({
        'message': all_messages,
        'label': all_labels
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_file = DATASET_DIR / 'text' / 'synthetic_dataset.csv'
    output_file.parent.mkdir(exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"Synthetic dataset created: {output_file}")
    print(f"Total samples: {len(df)}")
    print(f"Normal: {len(df[df['label'] == 'normal'])}")
    print(f"Sensitive: {len(df[df['label'] == 'sensitive'])}")
    
    return output_file

def main():
    """Main function to prepare all datasets"""
    print("=" * 60)
    print("Dataset Preparation for Sensitive Data Protection")
    print("=" * 60)
    
    # Prepare face detection dataset
    face_dir = prepare_face_dataset()
    
    # Prepare text classification dataset
    text_file = prepare_text_dataset()
    
    # Create synthetic dataset for testing
    synthetic_file = create_synthetic_dataset()
    
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nDataset directory: {DATASET_DIR}")
    print(f"Face dataset: {face_dir}")
    print(f"Text dataset: {text_file if text_file else 'Check instructions above'}")
    print(f"Synthetic dataset: {synthetic_file}")
    
    print("\nNext steps:")
    print("1. Download recommended datasets from Kaggle if needed")
    print("2. Run train_text_classifier.py to train text model")
    print("3. Run train_face_detector.py to train face detection (or use pretrained)")
    print("4. Start the web application with: python backend/app.py")

if __name__ == "__main__":
    main()