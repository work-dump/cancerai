#!/usr/bin/env python3
"""
Local Dataset Evaluation Script for Tricorder-3 Competition

This script allows you to evaluate your ONNX model against a local dataset
without needing to connect to HuggingFace.

Usage:
    python scripts/evaluate_local.py \
        --model_path your_model.onnx \
        --dataset_dir competition_datasets/tricorder-3-mainnet/extracted

The dataset directory should contain:
    - *.jpg images
    - labels.csv with columns: NewFileName, Class, Age, Location, Gender
"""

import os
import sys
import argparse
import csv
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict

# Tricorder-3 class mapping (11 classes, 0-indexed)
CLASS_NAMES = [
    "AKIEC",    # 0 - Actinic keratosis (Medium risk)
    "BCC",      # 1 - Basal cell carcinoma (Malignant)
    "BEN_OTH",  # 2 - Other benign (Benign)
    "BKL",      # 3 - Benign keratinocytic (Medium risk)
    "DF",       # 4 - Dermatofibroma (Benign)
    "INF",      # 5 - Inflammatory (Benign)
    "MAL_OTH",  # 6 - Other malignant (Malignant)
    "MEL",      # 7 - Melanoma (Malignant)
    "NV",       # 8 - Melanocytic Nevus (Benign)
    "SCCKA",    # 9 - Squamous cell carcinoma (Malignant)
    "VASC",     # 10 - Vascular lesions (Medium risk)
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Risk categories for weighted F1 calculation
MALIGNANT_CLASSES = [1, 6, 7, 9]  # BCC, MAL_OTH, MEL, SCCKA
MEDIUM_RISK_CLASSES = [0, 3, 10]  # AKIEC, BKL, VASC
BENIGN_CLASSES = [2, 4, 5, 8]     # BEN_OTH, DF, INF, NV

# Location mapping
LOCATION_MAP = {
    'arm': 1, 'feet': 2, 'genitalia': 3, 'hand': 4,
    'head': 5, 'leg': 6, 'torso': 7
}


def preprocess_image(image_path: str, target_size=(512, 512)) -> np.ndarray:
    """Preprocess image for model input."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    return img_array


def load_dataset(dataset_dir: str):
    """Load dataset from local directory."""
    dataset_path = Path(dataset_dir)
    
    # Find labels file
    labels_file = None
    for name in ['labels.csv', 'label.csv']:
        if (dataset_path / name).exists():
            labels_file = dataset_path / name
            break
    
    if not labels_file:
        raise FileNotFoundError(f"No labels.csv or label.csv found in {dataset_dir}")
    
    # Parse CSV
    entries = []
    with open(labels_file, 'r') as f:
        # Detect delimiter
        first_line = f.readline()
        f.seek(0)
        delimiter = ';' if ';' in first_line else ','
        
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            row_lower = {k.lower().strip(): v.strip().strip('"') for k, v in row.items()}
            
            filename = row_lower.get('newfilename') or row_lower.get('filepath') or row_lower.get('filename')
            label = row_lower.get('class') or row_lower.get('label')
            age = row_lower.get('age', '0')
            gender = row_lower.get('gender', 'm')
            location = row_lower.get('location', 'torso')
            
            if filename and label:
                image_path = dataset_path / filename
                if image_path.exists():
                    entries.append({
                        'image_path': str(image_path),
                        'label': label.upper(),
                        'age': int(age) if age.isdigit() else 30,
                        'gender': 1.0 if gender.lower() in ['m', 'male'] else 0.0,
                        'location': LOCATION_MAP.get(location.lower(), 7)
                    })
    
    return entries


def calculate_weighted_f1(y_true, y_pred):
    """Calculate weighted F1 score based on risk categories."""
    f1_scores = f1_score(y_true, y_pred, labels=list(range(11)), average=None, zero_division=0)
    
    # Calculate F1 for each category
    f1_malignant = np.mean([f1_scores[i] for i in MALIGNANT_CLASSES if i < len(f1_scores)])
    f1_medium = np.mean([f1_scores[i] for i in MEDIUM_RISK_CLASSES if i < len(f1_scores)])
    f1_benign = np.mean([f1_scores[i] for i in BENIGN_CLASSES if i < len(f1_scores)])
    
    # Weighted F1: (3 × malignant + 2 × medium + 1 × benign) / 6
    weighted_f1 = (3 * f1_malignant + 2 * f1_medium + 1 * f1_benign) / 6
    
    return weighted_f1, {
        'malignant': f1_malignant,
        'medium_risk': f1_medium,
        'benign': f1_benign
    }


def calculate_final_score(accuracy, weighted_f1, model_size_mb):
    """Calculate final competition score."""
    # Prediction score (90%)
    prediction_score = 0.5 * accuracy + 0.5 * weighted_f1
    
    # Efficiency score (10%)
    if model_size_mb <= 50:
        efficiency_score = 1.0
    elif model_size_mb >= 150:
        efficiency_score = 0.0
    else:
        efficiency_score = (150 - model_size_mb) / 100
    
    # Final score
    final_score = 0.9 * prediction_score + 0.1 * efficiency_score
    
    return final_score, prediction_score, efficiency_score


def main():
    parser = argparse.ArgumentParser(description='Evaluate ONNX model on local dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    args = parser.parse_args()
    
    # Check paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found at {args.dataset_dir}")
        sys.exit(1)
    
    # Get model size
    model_size_mb = os.path.getsize(args.model_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Dataset: {args.dataset_dir}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading ONNX model...")
    session = ort.InferenceSession(args.model_path)
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"Model inputs: {input_names}")
    
    # Load dataset
    print("Loading dataset...")
    entries = load_dataset(args.dataset_dir)
    print(f"Found {len(entries)} valid entries")
    
    if len(entries) == 0:
        print("Error: No valid entries found in dataset")
        sys.exit(1)
    
    # Count class distribution
    class_counts = defaultdict(int)
    for entry in entries:
        class_counts[entry['label']] += 1
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    
    # Run inference
    print("\nRunning inference...")
    y_true = []
    y_pred = []
    skipped = 0
    
    for i, entry in enumerate(entries):
        try:
            # Get true label
            if entry['label'] not in CLASS_TO_IDX:
                print(f"  Warning: Unknown class '{entry['label']}' for {entry['image_path']}, skipping")
                skipped += 1
                continue
            
            true_idx = CLASS_TO_IDX[entry['label']]
            
            # Preprocess image
            img = preprocess_image(entry['image_path'])
            img_batch = np.expand_dims(img, axis=0)
            
            # Prepare demographics
            demo = np.array([[entry['age'], entry['gender'], entry['location']]], dtype=np.float32)
            
            # Run inference
            if len(input_names) == 2:
                outputs = session.run(None, {input_names[0]: img_batch, input_names[1]: demo})
            else:
                outputs = session.run(None, {input_names[0]: img_batch})
            
            probs = outputs[0].flatten()
            pred_idx = np.argmax(probs)
            
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(entries)} images...")
                
        except Exception as e:
            print(f"  Error processing {entry['image_path']}: {e}")
            skipped += 1
    
    print(f"\nProcessed {len(y_true)} images, skipped {skipped}")
    
    if len(y_true) == 0:
        print("Error: No images were successfully processed")
        sys.exit(1)
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1, category_f1 = calculate_weighted_f1(y_true, y_pred)
    final_score, prediction_score, efficiency_score = calculate_final_score(accuracy, weighted_f1, model_size_mb)
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nMetrics:")
    print(f"  Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Weighted F1:  {weighted_f1:.4f}")
    print(f"\nF1 by Risk Category:")
    print(f"  Malignant (3x):    {category_f1['malignant']:.4f}")
    print(f"  Medium Risk (2x):  {category_f1['medium_risk']:.4f}")
    print(f"  Benign (1x):       {category_f1['benign']:.4f}")
    print(f"\nScoring:")
    print(f"  Prediction Score (90%):  {prediction_score:.4f}")
    print(f"  Efficiency Score (10%):  {efficiency_score:.4f}")
    print(f"\n  *** FINAL SCORE: {final_score:.4f} ***")
    print(f"{'='*60}")
    
    # Print confusion matrix
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(11)))
    
    # Print header
    print("        ", end="")
    for name in CLASS_NAMES:
        print(f"{name:>6}", end=" ")
    print()
    
    # Print rows
    for i, row in enumerate(cm):
        if i < len(CLASS_NAMES):
            print(f"{CLASS_NAMES[i]:>6}:", end=" ")
            for val in row:
                print(f"{val:>6}", end=" ")
            print()


if __name__ == "__main__":
    main()
