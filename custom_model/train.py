#!/usr/bin/env python3
"""
Complete Training Script for Tricorder-3 Competition

This script:
1. Loads configuration from train_config.yaml
2. Loads datasets from the specified folder
3. Trains the model using competition-optimized strategies
4. Evaluates using exact competition metrics
5. Exports the best model to ONNX format

Usage:
    # Use config file (recommended)
    python custom_model/train.py
    
    # Override specific settings
    python custom_model/train.py --epochs 100 --batch-size 32
    
    # Use different config file
    python custom_model/train.py --config my_config.yaml
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default train_config.yaml
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default config path (same directory as this script)
        config_path = Path(__file__).parent / "train_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Using default values.")
        return {}
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {config_path}")
        return config or {}
    except ImportError:
        print("PyYAML not installed. Install with: pip install pyyaml")
        print("Using default values.")
        return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default values.")
        return {}


def get_config_value(config: Dict, *keys, default=None):
    """Get nested config value safely."""
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value if value is not None else default

from custom_model import (
    # Model
    create_balanced_model,
    create_lightweight_model,
    create_larger_model,
    export_to_onnx,
    verify_onnx_model,
    Tricorder3Model,
    
    # Trainer
    Tricorder3Trainer,
    TrainingConfig,
    SkinLesionDataset,
    evaluate_predictions,
    get_train_transforms,
    get_val_transforms,
    print_class_distribution,
)


# ============================================================================
# Constants
# ============================================================================

NUM_CLASSES = 11

CLASS_NAMES = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]

# Mapping from various label formats to class index
LABEL_MAPPING = {
    # Short names
    "AKIEC": 0, "BCC": 1, "BEN_OTH": 2, "BKL": 3, "DF": 4,
    "INF": 5, "MAL_OTH": 6, "MEL": 7, "NV": 8, "SCCKA": 9, "VASC": 10,
    # Full names (lowercase)
    "actinic keratosis": 0, "basal cell carcinoma": 1, "benign": 2,
    "benign keratosis": 3, "dermatofibroma": 4, "inflammatory": 5,
    "malignant": 6, "melanoma": 7, "nevus": 8, "squamous cell": 9, "vascular": 10,
    # Numbers
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
}


# ============================================================================
# Data Loading
# ============================================================================

def find_labels_file(data_dir: str) -> Optional[str]:
    """Find labels CSV file in data directory."""
    candidates = [
        "labels.csv",
        "metadata.csv", 
        "data.csv",
        "dataset.csv",
        "train.csv",
    ]
    
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    
    # Search for any CSV
    for f in os.listdir(data_dir):
        if f.endswith('.csv'):
            return os.path.join(data_dir, f)
    
    return None


def is_presplit_dataset(data_dir: str) -> bool:
    """Check if the dataset directory contains pre-split train/val/test folders."""
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    
    # Check if train and val directories exist with labels.csv
    return (
        train_dir.exists() and 
        val_dir.exists() and
        (train_dir / "labels.csv").exists() and
        (val_dir / "labels.csv").exists()
    )


def load_split_folder(split_dir: Path) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """Load a single split folder (train, val, or test)."""
    labels_file = split_dir / "labels.csv"
    
    if not labels_file.exists():
        raise FileNotFoundError(f"No labels.csv found in {split_dir}")
    
    df = pd.read_csv(labels_file)
    
    image_paths = []
    labels = []
    metadata = []
    
    for _, row in df.iterrows():
        # Get image path (relative to split_dir)
        img_path = row.get('image', row.get('image_path', row.get('filename', '')))
        
        if not img_path:
            continue
        
        # Make path absolute or relative to split_dir
        full_path = split_dir / img_path
        
        # Handle symlinks - resolve to actual file
        if full_path.is_symlink():
            full_path = full_path.resolve()
        
        if not full_path.exists():
            # Try as absolute path
            if Path(img_path).exists():
                full_path = Path(img_path)
            else:
                print(f"Warning: Image not found: {img_path}")
                continue
        
        # Get label
        label_str = str(row.get('label', row.get('class', row.get('diagnosis', ''))))
        
        if label_str in LABEL_MAPPING:
            label_idx = LABEL_MAPPING[label_str]
        else:
            try:
                label_idx = int(label_str)
            except:
                print(f"Warning: Unknown label {label_str}")
                continue
        
        image_paths.append(str(full_path))
        labels.append(label_idx)
        
        # Metadata
        meta = {
            'age': row.get('age', row.get('age_approx', 50)),
            'sex': row.get('sex', row.get('gender', '')),
            'localization': row.get('localization', row.get('site', '')),
        }
        metadata.append(meta)
    
    return image_paths, labels, metadata


def load_presplit_dataset(data_dir: str) -> Dict[str, Tuple[List[str], List[int], List[Dict[str, Any]]]]:
    """
    Load a pre-split dataset with train/val/test folders.
    
    Returns:
        Dict with 'train', 'val', and optionally 'test' keys
    """
    data_path = Path(data_dir)
    
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        split_dir = data_path / split_name
        
        if split_dir.exists() and (split_dir / "labels.csv").exists():
            print(f"  Loading {split_name}...")
            paths, labels, meta = load_split_folder(split_dir)
            splits[split_name] = (paths, labels, meta)
            print(f"    Found {len(paths)} samples")
    
    return splits


def load_dataset(
    data_dir: str,
    labels_file: Optional[str] = None,
) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """
    Load dataset from directory.
    
    Supports multiple formats:
    1. CSV with columns: image, label, age, gender, location
    2. Directory with images and labels.csv
    3. Directory with class subfolders
    4. Pre-split dataset with train/val/test folders (returns train only)
    
    Returns:
        Tuple of (image_paths, labels, metadata)
    """
    data_dir = Path(data_dir)
    
    # Try to find labels file
    if labels_file is None:
        labels_file = find_labels_file(str(data_dir))
    
    image_paths = []
    labels = []
    metadata = []
    
    if labels_file and os.path.exists(labels_file):
        # Load from CSV
        print(f"Loading from CSV: {labels_file}")
        df = pd.read_csv(labels_file)
        
        # Find image column (case-insensitive)
        df.columns = [c.strip() for c in df.columns]  # Strip whitespace
        col_lower = {c.lower(): c for c in df.columns}
        
        img_col = None
        for col in ['newfilename', 'image', 'filename', 'file', 'image_id', 'id', 'name']:
            if col in col_lower:
                img_col = col_lower[col]
                break
        
        # Find label column
        label_col = None
        for col in ['class', 'label', 'diagnosis', 'target', 'category']:
            if col in col_lower:
                label_col = col_lower[col]
                break
        
        if img_col is None or label_col is None:
            print(f"Warning: Could not find image/label columns in CSV")
            print(f"Available columns: {df.columns.tolist()}")
            # Try to use first and second columns
            img_col = df.columns[0]
            label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        print(f"Using columns: image='{img_col}', label='{label_col}'")
        
        for _, row in df.iterrows():
            # Image path
            img_name = str(row[img_col])
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_name += '.jpg'
            
            img_path = data_dir / img_name
            if not img_path.exists():
                # Try in parent directory
                img_path = data_dir.parent / img_name
            if not img_path.exists():
                continue
            
            image_paths.append(str(img_path))
            
            # Label
            label = row[label_col]
            if isinstance(label, str):
                label = label.upper().strip()
                label = LABEL_MAPPING.get(label, LABEL_MAPPING.get(label.lower(), 0))
            labels.append(int(label))
            
            # Metadata (case-insensitive column lookup)
            meta = {}
            
            # Age
            age_col = col_lower.get('age')
            if age_col:
                age = row[age_col]
                meta['age'] = float(age) if pd.notna(age) else 50.0
            
            # Gender
            gender_col = col_lower.get('gender') or col_lower.get('sex')
            if gender_col:
                meta['gender'] = str(row[gender_col]) if pd.notna(row[gender_col]) else ''
            
            # Location
            loc_col = col_lower.get('location') or col_lower.get('localization')
            if loc_col:
                meta['location'] = str(row[loc_col]) if pd.notna(row[loc_col]) else ''
            
            metadata.append(meta)
    
    else:
        # Try class subfolders structure
        print(f"Looking for class subfolders in {data_dir}")
        for class_name in CLASS_NAMES:
            class_dir = data_dir / class_name
            if class_dir.exists():
                class_idx = CLASS_NAMES.index(class_name)
                for img_file in class_dir.glob("*.jpg"):
                    image_paths.append(str(img_file))
                    labels.append(class_idx)
                    metadata.append({})
                for img_file in class_dir.glob("*.png"):
                    image_paths.append(str(img_file))
                    labels.append(class_idx)
                    metadata.append({})
        
        # If no class folders, just load all images with dummy labels
        if not image_paths:
            print(f"No class folders found, loading all images with dummy labels")
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_file in data_dir.glob(ext):
                    image_paths.append(str(img_file))
                    labels.append(0)  # Dummy label
                    metadata.append({})
    
    print(f"Loaded {len(image_paths)} images")
    
    return image_paths, labels, metadata


def create_synthetic_dataset(
    num_samples: int = 1000,
    output_dir: str = "synthetic_data",
) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """
    Create synthetic dataset for testing the training pipeline.
    
    Creates random colored images with class-specific colors for visualization.
    """
    from PIL import Image
    import random
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Class-specific base colors (for visual debugging)
    class_colors = [
        (200, 150, 150),  # AKIEC - pinkish
        (100, 50, 50),    # BCC - dark red
        (200, 200, 150),  # BEN_OTH - beige
        (180, 150, 100),  # BKL - brownish
        (150, 100, 80),   # DF - tan
        (250, 200, 200),  # INF - light pink
        (80, 40, 40),     # MAL_OTH - dark brown
        (50, 30, 30),     # MEL - very dark
        (150, 130, 100),  # NV - medium brown
        (120, 60, 60),    # SCCKA - reddish brown
        (200, 150, 200),  # VASC - purplish
    ]
    
    image_paths = []
    labels = []
    metadata = []
    
    for i in range(num_samples):
        # Random class (weighted towards benign for realism)
        weights = [0.05, 0.08, 0.15, 0.08, 0.08, 0.08, 0.05, 0.08, 0.25, 0.05, 0.05]
        label = random.choices(range(11), weights=weights)[0]
        
        # Create image with class-specific color + noise
        base_color = class_colors[label]
        img_array = np.random.randint(
            low=max(0, base_color[0]-50),
            high=min(255, base_color[0]+50),
            size=(512, 512, 3),
            dtype=np.uint8
        )
        
        # Add some structure (circle in center)
        center = 256
        radius = random.randint(80, 150)
        y, x = np.ogrid[:512, :512]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        lesion_color = tuple(max(0, c - 30) for c in base_color)
        img_array[mask] = lesion_color
        
        # Save image
        img = Image.fromarray(img_array)
        img_path = os.path.join(output_dir, f"image_{i:05d}.jpg")
        img.save(img_path)
        
        image_paths.append(img_path)
        labels.append(label)
        
        # Random metadata
        metadata.append({
            'age': random.randint(20, 80),
            'gender': random.choice(['male', 'female']),
            'location': random.choice(['arm', 'leg', 'torso', 'head', 'hand']),
        })
    
    # Save labels CSV
    df = pd.DataFrame({
        'image': [os.path.basename(p) for p in image_paths],
        'label': [CLASS_NAMES[l] for l in labels],
        'age': [m['age'] for m in metadata],
        'gender': [m['gender'] for m in metadata],
        'location': [m['location'] for m in metadata],
    })
    df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)
    
    print(f"Created synthetic dataset with {num_samples} images in {output_dir}")
    
    return image_paths, labels, metadata


# ============================================================================
# Main Training Function
# ============================================================================

def train(args):
    """Main training function."""
    
    print("="*60)
    print("Tricorder-3 Competition Training")
    print("="*60)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load or create dataset
    if args.synthetic:
        print("\nCreating synthetic dataset for testing...")
        image_paths, labels, metadata = create_synthetic_dataset(
            num_samples=args.synthetic_samples,
            output_dir=args.synthetic_dir,
        )
        use_presplit = False
    else:
        print(f"\nLoading dataset from: {args.data_dir}")
        
        # Check if this is a pre-split dataset
        use_presplit = is_presplit_dataset(args.data_dir)
        
        if use_presplit:
            print("  ✓ Detected pre-split dataset (train/val/test folders)")
            splits = load_presplit_dataset(args.data_dir)
            
            if 'train' not in splits or 'val' not in splits:
                print("ERROR: Pre-split dataset must have train/ and val/ folders!")
                return
            
            train_paths, train_labels, train_meta = splits['train']
            val_paths, val_labels, val_meta = splits['val']
            
            # Use train data for distribution printing
            image_paths = train_paths
            labels = train_labels
            metadata = train_meta
        else:
            print("  → Using on-the-fly split (run split_dataset.py for reproducible splits)")
            image_paths, labels, metadata = load_dataset(
                args.data_dir,
                args.labels_file,
            )
    
    if len(image_paths) == 0:
        print("ERROR: No images found!")
        return
    
    # Print class distribution
    print_class_distribution(labels if not use_presplit else train_labels)
    
    # Split data (only if not pre-split)
    if not use_presplit:
        print(f"\nSplitting data (val_split={args.val_split})...")
        
        # Check if stratification is possible (each class needs at least 2 samples)
        class_counts = np.bincount(labels, minlength=NUM_CLASSES)
        can_stratify = all(c == 0 or c >= 2 for c in class_counts)
        
        try:
            train_paths, val_paths, train_labels, val_labels, train_meta, val_meta = train_test_split(
                image_paths, labels, metadata,
                test_size=args.val_split,
                stratify=labels if can_stratify and len(set(labels)) > 1 else None,
                random_state=args.seed,
            )
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}), using random split")
            train_paths, val_paths, train_labels, val_labels, train_meta, val_meta = train_test_split(
                image_paths, labels, metadata,
                test_size=args.val_split,
                stratify=None,
                random_state=args.seed,
            )
    
    print(f"\n  Train: {len(train_paths)} samples")
    print(f"  Val: {len(val_paths)} samples")
    if use_presplit and 'test' in splits:
        print(f"  Test: {len(splits['test'][0])} samples (held out)")
    
    # Create datasets
    train_dataset = SkinLesionDataset(
        train_paths, train_labels, train_meta,
        transform=get_train_transforms(),
    )
    
    val_dataset = SkinLesionDataset(
        val_paths, val_labels, val_meta,
        transform=get_val_transforms(),
    )
    
    # Create model
    print(f"\nCreating model: {args.model_type}")
    if args.model_type == "lightweight":
        model = create_lightweight_model()
    elif args.model_type == "balanced":
        model = create_balanced_model()
    elif args.model_type == "larger":
        model = create_larger_model()
    else:
        model = create_balanced_model()
    
    params = model.count_parameters()
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Estimated size: {model.get_model_size_mb():.2f} MB")
    
    # Training config
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        save_dir=args.save_dir,
        save_every_n_epochs=args.save_every,
        num_workers=args.num_workers,
        use_tensorboard=args.tensorboard,
        tensorboard_dir=args.tensorboard_dir,
    )
    
    # Create trainer
    trainer = Tricorder3Trainer(model, config)
    
    # Train
    print("\nStarting training...")
    history = trainer.fit(
        train_dataset,
        val_dataset,
        use_weighted_sampling=args.weighted_sampling,
        resume_from=args.resume,
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    final_metrics = trainer.validate(
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )
    )
    
    print(f"\nFinal Validation Results:")
    print(f"  Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"  Weighted F1: {final_metrics['val_weighted_f1']:.4f}")
    print(f"  F1 HIGH_RISK: {final_metrics['val_f1_high_risk']:.4f}")
    print(f"  F1 MEDIUM_RISK: {final_metrics['val_f1_medium_risk']:.4f}")
    print(f"  F1 BENIGN: {final_metrics['val_f1_benign']:.4f}")
    print(f"  Prediction Score: {final_metrics['val_prediction_score']:.4f}")
    print(f"  Competition Score: {final_metrics['val_competition_score']:.4f}")
    
    # Export to ONNX
    if args.export_onnx:
        print(f"\nExporting model to ONNX...")
        onnx_path = os.path.join(args.save_dir, "model.onnx")
        export_to_onnx(model, onnx_path)
        
        # Verify
        print("Verifying ONNX model...")
        verify_onnx_model(onnx_path)
        
        # Check size
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        efficiency = 1.0 if size_mb <= 50 else max(0, (150 - size_mb) / 100)
        print(f"\nONNX Model:")
        print(f"  Path: {onnx_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Efficiency Score: {efficiency:.4f}")
    
    # Save training history
    history_path = os.path.join(args.save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        # Convert numpy values to Python types
        history_clean = []
        for h in history:
            h_clean = {}
            for k, v in h.items():
                if isinstance(v, np.ndarray):
                    h_clean[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    h_clean[k] = float(v)
                else:
                    h_clean[k] = v
            history_clean.append(h_clean)
        json.dump(history_clean, f, indent=2)
    print(f"Saved training history to: {history_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    # First, parse just the --config argument to load config file
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    pre_args, _ = pre_parser.parse_known_args()
    
    # Load configuration
    config = load_config(pre_args.config)
    
    # Main parser with config-based defaults
    parser = argparse.ArgumentParser(
        description="Train Tricorder-3 Competition Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file (default: custom_model/train_config.yaml)"
    )
    
    # Data
    parser.add_argument(
        "--data-dir", type=str, 
        default=get_config_value(config, 'data', 'data_dir', default="training_data"),
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--labels-file", type=str, 
        default=get_config_value(config, 'data', 'labels_file', default=None),
        help="Path to labels CSV file (auto-detected if not specified)"
    )
    parser.add_argument(
        "--val-split", type=float, 
        default=get_config_value(config, 'data', 'val_split', default=0.2),
        help="Validation split ratio"
    )
    
    # Synthetic data (for testing)
    parser.add_argument(
        "--synthetic", action="store_true",
        default=get_config_value(config, 'synthetic', 'enabled', default=False),
        help="Use synthetic data for testing"
    )
    parser.add_argument(
        "--synthetic-samples", type=int, 
        default=get_config_value(config, 'synthetic', 'num_samples', default=500),
        help="Number of synthetic samples to create"
    )
    parser.add_argument(
        "--synthetic-dir", type=str, 
        default=get_config_value(config, 'synthetic', 'output_dir', default="synthetic_data"),
        help="Directory for synthetic data"
    )
    
    # Model
    parser.add_argument(
        "--model-type", type=str, 
        default=get_config_value(config, 'model', 'type', default="balanced"),
        choices=["lightweight", "balanced", "larger"],
        help="Model type to use"
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Augmentation
    parser.add_argument("--use-mixup", action="store_true", default=True)
    parser.add_argument("--no-mixup", dest="use_mixup", action="store_false")
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--weighted-sampling", action="store_true", default=True)
    parser.add_argument("--no-weighted-sampling", dest="weighted_sampling", action="store_false")
    
    # Output
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--save-every", type=int, default=5,
        help="Save checkpoint every N epochs (0 = only save best)"
    )
    parser.add_argument("--export-onnx", action="store_true", default=True)
    parser.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
    
    # Resume training
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from (e.g., checkpoints/latest.pt)"
    )
    
    # TensorBoard
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    parser.add_argument("--tensorboard-dir", type=str, default="runs")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
