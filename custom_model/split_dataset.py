#!/usr/bin/env python3
"""
Dataset Split Script for Tricorder-3 Competition

Splits a downloaded dataset into train/val/test folders for reproducible training.
This ensures the same split is used across training sessions and resume.

Usage:
    # Split the unified labels_final.csv into train/val/test
    python custom_model/split_dataset.py --input training_data --output training_data_split
    
    # Custom split ratios (default: 80% train, 10% val, 10% test)
    python custom_model/split_dataset.py --input training_data --output training_data_split \
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
    
    # Split specific dataset folder
    python custom_model/split_dataset.py --input training_data/isic_milk10k --output milk10k_split
    
    # Use stratified split (recommended for imbalanced datasets)
    python custom_model/split_dataset.py --input training_data --output training_data_split --stratify

After splitting, train with:
    python custom_model/train.py --data-dir training_data_split
"""

import os
import sys
import argparse
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math


def safe_get(row, keys, default):
    """
    Safely get a value from a pandas row, handling NaN values.
    
    Args:
        row: pandas Series (row from DataFrame)
        keys: str or list of str - column name(s) to try
        default: default value if all keys are missing or NaN
        
    Returns:
        The value or default if NaN/missing
    """
    if isinstance(keys, str):
        keys = [keys]
    
    for key in keys:
        if key in row.index:
            val = row[key]
            # Check for NaN (works for both numeric and string NaN)
            if pd.notna(val):
                return val
    
    return default


def find_labels_file(data_dir: Path) -> Optional[Path]:
    """Find labels CSV file in directory."""
    candidates = [
        "labels_final.csv",
        "labels.csv",
        "metadata.csv",
        "train_labels.csv",
    ]
    
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    
    return None


def is_milk10k_format(data_dir: Path) -> bool:
    """Check if directory contains MILK10k format files."""
    ground_truth = data_dir / "MILK10k_Training_GroundTruth.csv"
    metadata = data_dir / "MILK10k_Training_Metadata.csv"
    return ground_truth.exists() and metadata.exists()


def load_milk10k_dataset(data_dir: Path) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Load MILK10k dataset from its specific format.
    
    MILK10k structure:
    - MILK10k_Training_GroundTruth.csv: lesion_id + one-hot labels
    - MILK10k_Training_Metadata.csv: lesion_id, image_type, isic_id, demographics
    - MILK10k_Training_Input/IL_xxx/ISIC_xxx.jpg: images in nested folders
    """
    print("  Detected MILK10k format")
    
    # Class names (Tricorder-3 classes)
    CLASS_NAMES = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']
    
    # Load ground truth (one-hot labels)
    gt_path = data_dir / "MILK10k_Training_GroundTruth.csv"
    gt_df = pd.read_csv(gt_path)
    print(f"  Ground truth: {len(gt_df)} lesions")
    
    # Map lesion_id to label
    lesion_labels = {}
    for _, row in gt_df.iterrows():
        lesion_id = row['lesion_id']
        # Find which class has 1.0
        for cls in CLASS_NAMES:
            if cls in row and row[cls] == 1.0:
                lesion_labels[lesion_id] = cls
                break
    
    # Load metadata
    meta_path = data_dir / "MILK10k_Training_Metadata.csv"
    meta_df = pd.read_csv(meta_path)
    print(f"  Metadata: {len(meta_df)} image entries")
    
    # Find all images
    image_dirs = [
        data_dir / "MILK10k_Training_Input",
        data_dir / "images",
        data_dir,
    ]
    
    all_images = {}
    for img_dir in image_dirs:
        if img_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
                for img_path in img_dir.rglob(ext):
                    # Extract ISIC ID from filename
                    isic_id = img_path.stem  # e.g., "ISIC_4671410"
                    all_images[isic_id] = img_path
    
    print(f"  Found {len(all_images)} images")
    
    if not all_images:
        # Try to extract zip if exists
        zip_path = data_dir / "MILK10k_Training_Input.zip"
        if zip_path.exists():
            import zipfile
            print(f"  Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(data_dir)
            # Retry finding images
            for img_dir in image_dirs:
                if img_dir.exists():
                    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
                        for img_path in img_dir.rglob(ext):
                            isic_id = img_path.stem
                            all_images[isic_id] = img_path
            print(f"  Found {len(all_images)} images after extraction")
    
    # Process all images (both clinical and dermoscopic)
    image_paths = []
    labels = []
    metadata = []
    
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="  Processing"):
        lesion_id = row.get('lesion_id', '')
        isic_id = row.get('isic_id', '')
        
        # Get label
        label = lesion_labels.get(lesion_id)
        if label is None:
            continue
        
        # Get image path
        img_path = all_images.get(isic_id)
        if img_path is None:
            continue
        
        image_paths.append(str(img_path))
        labels.append(label)
        
        # Extract metadata (handle NaN values properly)
        meta = {
            'age': safe_get(row, ['age_approx', 'age'], 50),
            'sex': safe_get(row, ['sex', 'gender'], ''),
            'localization': safe_get(row, ['site', 'localization'], ''),
        }
        metadata.append(meta)
    
    print(f"  Loaded {len(image_paths)} samples")
    
    # Print class distribution
    label_counts = defaultdict(int)
    for lbl in labels:
        label_counts[lbl] += 1
    print("  Class distribution:")
    for cls in sorted(label_counts.keys()):
        print(f"    {cls}: {label_counts[cls]}")
    
    return image_paths, labels, metadata


def load_dataset_info(data_dir: Path) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Load dataset information from a directory.
    
    Supports multiple formats:
    1. labels_final.csv (from download_datasets.py)
    2. MILK10k format (MILK10k_Training_GroundTruth.csv + MILK10k_Training_Metadata.csv)
    3. Folder structure (class_name/image.jpg)
    
    Returns:
        Tuple of (image_paths, labels, metadata)
    """
    data_dir = Path(data_dir)
    
    # Check for MILK10k format first
    if is_milk10k_format(data_dir):
        return load_milk10k_dataset(data_dir)
    
    # Try to find standard labels file
    labels_file = find_labels_file(data_dir)
    
    if labels_file:
        print(f"Found labels file: {labels_file}")
        df = pd.read_csv(labels_file)
        
        # Detect columns
        image_col = None
        label_col = None
        
        for col in ['image', 'image_path', 'filename', 'path']:
            if col in df.columns:
                image_col = col
                break
        
        for col in ['label', 'class', 'diagnosis', 'category']:
            if col in df.columns:
                label_col = col
                break
        
        if image_col is None or label_col is None:
            raise ValueError(f"Could not detect image/label columns in {labels_file}")
        
        image_paths = df[image_col].tolist()
        labels = df[label_col].tolist()
        
        # Extract metadata (handle NaN values properly)
        metadata = []
        for _, row in df.iterrows():
            meta = {
                'age': safe_get(row, ['age', 'age_approx'], 50),
                'sex': safe_get(row, ['sex', 'gender'], ''),
                'localization': safe_get(row, ['localization', 'site'], ''),
            }
            metadata.append(meta)
        
        return image_paths, labels, metadata
    
    else:
        # Look for folder structure (class_name/image.jpg)
        print("No labels file found, looking for folder structure...")
        
        image_paths = []
        labels = []
        metadata = []
        
        for class_dir in sorted(data_dir.iterdir()):
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                class_name = class_dir.name
                
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        image_paths.append(str(img_path))
                        labels.append(class_name)
                        metadata.append({'age': 50, 'sex': '', 'localization': ''})
        
        if not image_paths:
            raise ValueError(f"No images found in {data_dir}")
        
        return image_paths, labels, metadata


def split_dataset(
    image_paths: List[str],
    labels: List[str],
    metadata: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify: bool = True,
    seed: int = 42,
) -> Dict[str, Tuple[List, List, List]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        metadata: List of metadata dicts
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        stratify: Whether to stratify by label
        seed: Random seed
        
    Returns:
        Dict with 'train', 'val', 'test' keys, each containing (paths, labels, metadata)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    np.random.seed(seed)
    random.seed(seed)
    
    indices = list(range(len(image_paths)))
    
    # Check if stratification is possible
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    
    min_count = min(label_counts.values())
    can_stratify = stratify and min_count >= 3
    
    if stratify and not can_stratify:
        print(f"⚠ Cannot stratify: some classes have fewer than 3 samples")
        print(f"  Falling back to random split")
    
    stratify_labels = labels if can_stratify else None
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_test_ratio,
        stratify=stratify_labels,
        random_state=seed,
    )
    
    # Second split: val vs test
    temp_labels = [labels[i] for i in temp_idx] if can_stratify else None
    relative_test_ratio = test_ratio / val_test_ratio
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=seed,
    )
    
    # Create split datasets
    splits = {
        'train': (
            [image_paths[i] for i in train_idx],
            [labels[i] for i in train_idx],
            [metadata[i] for i in train_idx],
        ),
        'val': (
            [image_paths[i] for i in val_idx],
            [labels[i] for i in val_idx],
            [metadata[i] for i in val_idx],
        ),
        'test': (
            [image_paths[i] for i in test_idx],
            [labels[i] for i in test_idx],
            [metadata[i] for i in test_idx],
        ),
    }
    
    return splits


def balance_split(
    paths: List[str],
    labels: List[str],
    metadata: List[Dict],
    max_per_class: int = 3000,
    min_per_class: int = 500,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Balance a dataset split by oversampling rare classes and undersampling common classes.
    
    Args:
        paths: List of image paths
        labels: List of labels
        metadata: List of metadata dicts
        max_per_class: Maximum samples per class (undersample if more)
        min_per_class: Minimum samples per class (oversample if less)
        seed: Random seed
        
    Returns:
        Tuple of (balanced_paths, balanced_labels, balanced_metadata)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Group by class
    class_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_indices[label].append(i)
    
    balanced_indices = []
    
    print(f"\n⚖️ Balancing (min={min_per_class}, max={max_per_class})...")
    
    for label in sorted(class_indices.keys()):
        indices = class_indices[label]
        count = len(indices)
        
        if count > max_per_class:
            # Undersample
            sampled = random.sample(indices, max_per_class)
            balanced_indices.extend(sampled)
            print(f"  {label}: {count} → {max_per_class} (undersampled)")
        elif count < min_per_class:
            # Oversample with replacement
            sampled = indices.copy()
            while len(sampled) < min_per_class:
                sampled.extend(random.choices(indices, k=min(min_per_class - len(sampled), len(indices))))
            balanced_indices.extend(sampled[:min_per_class])
            print(f"  {label}: {count} → {min_per_class} (oversampled)")
        else:
            balanced_indices.extend(indices)
            print(f"  {label}: {count}")
    
    # Shuffle
    random.shuffle(balanced_indices)
    
    balanced_paths = [paths[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]
    balanced_metadata = [metadata[i] for i in balanced_indices]
    
    return balanced_paths, balanced_labels, balanced_metadata


def save_split(
    splits: Dict[str, Tuple[List, List, List]],
    output_dir: Path,
    copy_images: bool = False,
    symlink_images: bool = True,
) -> None:
    """
    Save split dataset to output directory.
    
    Creates:
        output_dir/
        ├── train/
        │   ├── labels.csv
        │   └── images/ (symlinks or copies)
        ├── val/
        │   ├── labels.csv
        │   └── images/
        └── test/
            ├── labels.csv
            └── images/
    
    Args:
        splits: Dict from split_dataset()
        output_dir: Output directory
        copy_images: Whether to copy images (slower, uses more space)
        symlink_images: Whether to create symlinks (faster, saves space)
    """
    import platform
    
    output_dir = Path(output_dir)
    
    # Windows symlink warning
    is_windows = platform.system() == "Windows"
    if is_windows and symlink_images and not copy_images:
        print("\n⚠️  WARNING: Windows detected!")
        print("   Symlinks require Administrator privileges or Developer Mode.")
        print("   If training fails, re-run with: --copy-images")
        print("")
        # Try to test if symlinks work
        test_symlink = output_dir / ".symlink_test"
        test_target = output_dir / ".symlink_target"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            test_target.touch()
            test_symlink.symlink_to(test_target)
            test_symlink.unlink()
            test_target.unlink()
            print("   ✓ Symlink test passed - symlinks should work.\n")
        except OSError as e:
            print(f"   ❌ Symlink test FAILED: {e}")
            print("   → Automatically switching to copy mode.\n")
            copy_images = True
            symlink_images = False
            if test_target.exists():
                test_target.unlink()
    
    symlink_failures = 0
    
    for split_name, (paths, labels, metadata) in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = split_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Create labels CSV
        records = []
        
        for i, (path, label, meta) in enumerate(tqdm(
            zip(paths, labels, metadata),
            total=len(paths),
            desc=f"Processing {split_name}",
        )):
            src_path = Path(path)
            
            if not src_path.is_absolute():
                # Try to find the file
                if not src_path.exists():
                    # Try relative to current directory
                    src_path = Path.cwd() / path
            
            if not src_path.exists():
                print(f"⚠ Image not found: {path}")
                continue
            
            # Determine destination path
            dst_filename = f"{i:06d}_{src_path.name}"
            dst_path = images_dir / dst_filename
            
            # Copy or symlink
            if copy_images:
                shutil.copy2(src_path, dst_path)
            elif symlink_images:
                # Create symlink with absolute path
                try:
                    if dst_path.exists() or dst_path.is_symlink():
                        dst_path.unlink()
                    dst_path.symlink_to(src_path.resolve())
                except OSError as e:
                    # Symlink failed - fall back to copy
                    symlink_failures += 1
                    if symlink_failures == 1:
                        print(f"\n⚠️  Symlink failed, falling back to copy: {e}")
                    shutil.copy2(src_path, dst_path)
            
            # Record for CSV (ensure no NaN values in output)
            age_val = meta.get('age', 50)
            sex_val = meta.get('sex', '')
            loc_val = meta.get('localization', '')
            
            # Handle any remaining NaN values
            if pd.isna(age_val):
                age_val = 50
            if pd.isna(sex_val):
                sex_val = ''
            if pd.isna(loc_val):
                loc_val = ''
            
            records.append({
                'image': f"images/{dst_filename}",
                'label': label,
                'age': age_val,
                'sex': sex_val,
                'localization': loc_val,
            })
        
        # Save labels CSV
        df = pd.DataFrame(records)
        labels_path = split_dir / "labels.csv"
        df.to_csv(labels_path, index=False)
        
        print(f"  ✓ {split_name}: {len(records)} samples -> {split_dir}")
    
    if symlink_failures > 0:
        print(f"\n⚠️  {symlink_failures} symlinks failed and were copied instead.")
        print("   For future runs, use: --copy-images")


def print_split_stats(splits: Dict[str, Tuple[List, List, List]]) -> None:
    """Print statistics about the split."""
    print("\n" + "="*60)
    print("DATASET SPLIT STATISTICS")
    print("="*60)
    
    total = sum(len(s[0]) for s in splits.values())
    
    for split_name, (paths, labels, _) in splits.items():
        print(f"\n{split_name.upper()}: {len(paths)} samples ({100*len(paths)/total:.1f}%)")
        
        # Class distribution
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        
        print(f"  Classes: {len(label_counts)}")
        for label, count in sorted(label_counts.items()):
            pct = 100 * count / len(labels)
            print(f"    {label}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input dataset directory (e.g., training_data)"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output directory for split dataset"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Ratio for training set"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Ratio for validation set"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1,
        help="Ratio for test set"
    )
    parser.add_argument(
        "--stratify", action="store_true", default=True,
        help="Use stratified split (recommended)"
    )
    parser.add_argument(
        "--no-stratify", dest="stratify", action="store_false",
        help="Disable stratified split"
    )
    parser.add_argument(
        "--copy-images", action="store_true",
        help="Copy images instead of creating symlinks"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    # Balancing options
    parser.add_argument(
        "--balance", action="store_true",
        help="Balance the training set by over/undersampling"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=3000,
        help="Maximum samples per class when balancing"
    )
    parser.add_argument(
        "--min-per-class", type=int, default=500,
        help="Minimum samples per class when balancing"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    print("="*60)
    print("DATASET SPLITTER")
    print("="*60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Stratify: {args.stratify}")
    print(f"Balance training set: {args.balance}")
    if args.balance:
        print(f"  Max per class: {args.max_per_class}")
        print(f"  Min per class: {args.min_per_class}")
    print(f"Seed: {args.seed}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    image_paths, labels, metadata = load_dataset_info(input_dir)
    print(f"  Found {len(image_paths)} samples")
    
    # Split
    print("\n✂️ Splitting dataset...")
    splits = split_dataset(
        image_paths, labels, metadata,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify=args.stratify,
        seed=args.seed,
    )
    
    # Balance training set only (not val/test to keep evaluation fair)
    if args.balance:
        train_paths, train_labels, train_meta = splits['train']
        train_paths, train_labels, train_meta = balance_split(
            train_paths, train_labels, train_meta,
            max_per_class=args.max_per_class,
            min_per_class=args.min_per_class,
            seed=args.seed,
        )
        splits['train'] = (train_paths, train_labels, train_meta)
        print(f"\n  ✓ Training set balanced: {len(train_paths)} samples")
    
    # Print stats
    print_split_stats(splits)
    
    # Save
    print("\n💾 Saving split dataset...")
    save_split(
        splits, output_dir,
        copy_images=args.copy_images,
        symlink_images=not args.copy_images,
    )
    
    print("\n" + "="*60)
    print("✅ SPLIT COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"  ├── train/  ({len(splits['train'][0])} samples)")
    print(f"  ├── val/    ({len(splits['val'][0])} samples)")
    print(f"  └── test/   ({len(splits['test'][0])} samples)")
    
    print(f"\n📝 Next steps:")
    print(f"  1. Train: python custom_model/train.py --data-dir {output_dir}")
    print(f"  2. Evaluate: python custom_model/evaluate.py --model checkpoints/model.onnx --data-dir {output_dir}/test")


if __name__ == "__main__":
    main()
