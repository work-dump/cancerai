#!/usr/bin/env python3
"""
Model Evaluation Script for Tricorder-3 Competition

Evaluates models using exact competition scoring:
- Accuracy (50% of prediction score)
- Weighted F1 (50% of prediction score)
- Efficiency based on model size

Supports both PyTorch (.pt) and ONNX (.onnx) models.

Usage:
    # Evaluate ONNX model
    python custom_model/evaluate.py --model checkpoints/model.onnx --data-dir competition_datasets/tricorder-3-mainnet/extracted
    
    # Evaluate PyTorch checkpoint
    python custom_model/evaluate.py --model checkpoints/best_model.pt --data-dir competition_datasets/tricorder-3-mainnet/extracted
    
    # Compare with existing skin.onnx model
    python custom_model/evaluate.py --model available_models/skin.onnx --data-dir competition_datasets/tricorder-3-mainnet/extracted
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_model.trainer import (
    evaluate_predictions,
    calculate_competition_score,
    calculate_weighted_f1,
    calculate_risk_category_f1,
    CLASS_NAMES,
    CLASS_RISK,
    HIGH_RISK_CLASSES,
    MEDIUM_RISK_CLASSES,
    BENIGN_CLASSES,
    NUM_CLASSES,
)

from custom_model.train import load_dataset, LABEL_MAPPING


# ============================================================================
# Model Loading
# ============================================================================

class ModelWrapper:
    """Wrapper for unified inference across PyTorch and ONNX models."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_type = self._detect_model_type()
        self.model = None
        self.session = None
        self._load_model()
    
    def _detect_model_type(self) -> str:
        if self.model_path.endswith('.onnx'):
            return 'onnx'
        elif self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
            return 'pytorch'
        else:
            raise ValueError(f"Unknown model type: {self.model_path}")
    
    def _load_model(self):
        if self.model_type == 'onnx':
            import onnxruntime as ort
            self.session = ort.InferenceSession(self.model_path)
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            print(f"Loaded ONNX model: {self.model_path}")
            print(f"  Inputs: {self.input_names}")
            print(f"  Outputs: {self.output_names}")
        else:
            from custom_model import create_balanced_model
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model = create_balanced_model()
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"Loaded PyTorch model: {self.model_path}")
    
    def get_size_mb(self) -> float:
        return os.path.getsize(self.model_path) / (1024 * 1024)
    
    def predict(
        self, 
        image: np.ndarray, 
        demographics: np.ndarray
    ) -> np.ndarray:
        """
        Run inference on a single sample.
        
        Args:
            image: (1, 3, 512, 512) float32 array, normalized [0, 1]
            demographics: (1, 3) float32 array [age, gender, location]
        
        Returns:
            (1, 11) probability array
        """
        if self.model_type == 'onnx':
            # Prepare inputs
            inputs = {
                self.input_names[0]: image.astype(np.float32),
                self.input_names[1]: demographics.astype(np.float32),
            }
            
            # Run inference
            outputs = self.session.run(None, inputs)
            probs = outputs[0]
            
            # Apply softmax if output is logits (sum != 1)
            if abs(probs.sum() - 1.0) > 0.1:
                probs = self._softmax(probs)
            
            return probs
        else:
            with torch.no_grad():
                image_tensor = torch.from_numpy(image)
                demo_tensor = torch.from_numpy(demographics)
                probs = self.model(image_tensor, demo_tensor)
                return probs.numpy()
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)


# ============================================================================
# Image Preprocessing
# ============================================================================

def preprocess_image(
    image_path: str, 
    target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Returns:
        (1, 3, H, W) float32 array, normalized to [0, 1]
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Transpose to (C, H, W) and add batch dimension
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def encode_demographics(
    age: float = 50.0,
    gender: str = '',
    location: str = '',
) -> np.ndarray:
    """
    Encode demographics to model input format.
    
    Returns:
        (1, 3) float32 array [age, gender_code, location_code]
    """
    # Gender encoding
    gender = str(gender).lower()
    if gender in ['male', 'm', '1']:
        gender_code = 1.0
    elif gender in ['female', 'f', '0']:
        gender_code = 0.0
    else:
        gender_code = -1.0
    
    # Location encoding
    location_map = {
        'arm': 1, 'feet': 2, 'genitalia': 3, 'hand': 4,
        'head': 5, 'leg': 6, 'torso': 7,
    }
    location = str(location).lower()
    location_code = float(location_map.get(location, 0))
    
    return np.array([[age, gender_code, location_code]], dtype=np.float32)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(
    model: ModelWrapper,
    image_paths: List[str],
    labels: List[int],
    metadata: List[Dict[str, Any]],
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate model on dataset.
    
    Returns:
        Dictionary with all competition metrics
    """
    all_preds = []
    all_probs = []
    all_labels = []
    
    print(f"\nEvaluating on {len(image_paths)} samples...")
    
    for i, (img_path, label, meta) in enumerate(tqdm(
        zip(image_paths, labels, metadata), 
        total=len(image_paths),
        desc="Evaluating"
    )):
        try:
            # Preprocess
            image = preprocess_image(img_path)
            demographics = encode_demographics(
                age=meta.get('age', 50),
                gender=meta.get('gender', ''),
                location=meta.get('location', ''),
            )
            
            # Predict
            probs = model.predict(image, demographics)
            pred = np.argmax(probs, axis=-1)[0]
            
            all_preds.append(pred)
            all_probs.append(probs[0])
            all_labels.append(label)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Convert to numpy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Calculate metrics
    model_size = model.get_size_mb()
    efficiency = 1.0 if model_size <= 50 else max(0, (150 - model_size) / 100)
    
    metrics = evaluate_predictions(y_true, y_pred, efficiency)
    
    # Add additional details
    metrics['model_size_mb'] = model_size
    metrics['num_samples'] = len(y_true)
    metrics['predictions'] = y_probs.tolist()
    
    return metrics


def print_evaluation_results(metrics: Dict[str, Any], model_path: str):
    """Pretty print evaluation results."""
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nModel: {model_path}")
    print(f"Size: {metrics['model_size_mb']:.2f} MB")
    print(f"Samples Evaluated: {metrics['num_samples']}")
    
    print("\n" + "-"*70)
    print("COMPETITION METRICS")
    print("-"*70)
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-"*45)
    print(f"{'Accuracy':<30} {metrics['accuracy']:>15.4f}")
    print(f"{'Weighted F1':<30} {metrics['weighted_f1']:>15.4f}")
    print(f"{'Prediction Score':<30} {metrics['prediction_score']:>15.4f}")
    print(f"{'Efficiency Score':<30} {metrics['efficiency_score']:>15.4f}")
    print("-"*45)
    print(f"{'COMPETITION SCORE':<30} {metrics['final_score']:>15.4f}")
    
    print("\n" + "-"*70)
    print("F1 SCORES BY RISK CATEGORY")
    print("-"*70)
    
    print(f"\n{'Category':<20} {'F1 Score':>15} {'Weight':>10}")
    print("-"*45)
    for cat, f1 in metrics['category_f1'].items():
        weight = {'HIGH_RISK': '3×', 'MEDIUM_RISK': '2×', 'BENIGN': '1×'}[cat]
        print(f"{cat:<20} {f1:>15.4f} {weight:>10}")
    
    print("\n" + "-"*70)
    print("F1 SCORES BY CLASS")
    print("-"*70)
    
    print(f"\n{'Class':<12} {'Risk':<10} {'F1 Score':>12}")
    print("-"*35)
    for i, f1 in enumerate(metrics['f1_by_class']):
        risk = CLASS_RISK[i]
        print(f"{CLASS_NAMES[i]:<12} {risk:<10} {f1:>12.4f}")
    
    print("\n" + "="*70)


def compare_models(
    model_paths: List[str],
    image_paths: List[str],
    labels: List[int],
    metadata: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Compare multiple models and return results as DataFrame."""
    
    results = []
    
    for model_path in model_paths:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_path}")
        print("="*70)
        
        try:
            model = ModelWrapper(model_path)
            metrics = evaluate_model(model, image_paths, labels, metadata)
            
            results.append({
                'model': os.path.basename(model_path),
                'size_mb': metrics['model_size_mb'],
                'accuracy': metrics['accuracy'],
                'weighted_f1': metrics['weighted_f1'],
                'f1_high_risk': metrics['category_f1']['HIGH_RISK'],
                'f1_medium_risk': metrics['category_f1']['MEDIUM_RISK'],
                'f1_benign': metrics['category_f1']['BENIGN'],
                'prediction_score': metrics['prediction_score'],
                'efficiency_score': metrics['efficiency_score'],
                'competition_score': metrics['final_score'],
            })
            
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")
            continue
    
    df = pd.DataFrame(results)
    df = df.sort_values('competition_score', ascending=False)
    
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Tricorder-3 models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to model file (.onnx or .pt)"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--labels-file", type=str, default=None,
        help="Path to labels CSV file"
    )
    parser.add_argument(
        "--compare", type=str, nargs='+', default=None,
        help="Additional models to compare"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for results JSON"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to evaluate"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    image_paths, labels, metadata = load_dataset(args.data_dir, args.labels_file)
    
    if len(image_paths) == 0:
        print("ERROR: No images found!")
        return
    
    # Limit samples if specified
    if args.max_samples and len(image_paths) > args.max_samples:
        indices = np.random.choice(len(image_paths), args.max_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
        metadata = [metadata[i] for i in indices]
        print(f"Limited to {args.max_samples} samples")
    
    print(f"Dataset: {len(image_paths)} samples")
    
    # Compare multiple models if specified
    if args.compare:
        all_models = [args.model] + args.compare
        comparison_df = compare_models(all_models, image_paths, labels, metadata)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        if args.output:
            comparison_df.to_csv(args.output.replace('.json', '.csv'), index=False)
            print(f"\nSaved comparison to: {args.output.replace('.json', '.csv')}")
    
    else:
        # Evaluate single model
        model = ModelWrapper(args.model)
        metrics = evaluate_model(model, image_paths, labels, metadata)
        print_evaluation_results(metrics, args.model)
        
        # Save results
        if args.output:
            # Remove non-serializable items
            metrics_save = {k: v for k, v in metrics.items() if k != 'predictions'}
            with open(args.output, 'w') as f:
                json.dump(metrics_save, f, indent=2)
            print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
