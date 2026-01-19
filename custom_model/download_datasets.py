#!/usr/bin/env python3
"""
Dataset Download and Preparation Script for Tricorder-3 Competition

This script downloads multiple skin lesion datasets and prepares them
for training a high-scoring competition model.

Datasets:
1. HAM10000 - 10,015 images, 7 classes
2. ISIC 2019 - 25,331 images, 9 classes  
3. Additional HuggingFace datasets

All datasets are mapped to Tricorder-3's 11 classes:
0: AKIEC, 1: BCC, 2: BEN_OTH, 3: BKL, 4: DF,
5: INF, 6: MAL_OTH, 7: MEL, 8: NV, 9: SCCKA, 10: VASC

Usage:
    python custom_model/download_datasets.py --output-dir training_data
    python custom_model/download_datasets.py --output-dir training_data --skip-large
"""

import os
import sys
import json
import shutil
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import urllib.request
import zipfile
import tarfile

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ============================================================================
# Tricorder-3 Class Mapping
# ============================================================================

TRICORDER_CLASSES = [
    "AKIEC",    # 0: Actinic keratosis/intraepidermal carcinoma
    "BCC",      # 1: Basal cell carcinoma
    "BEN_OTH",  # 2: Other benign proliferations
    "BKL",      # 3: Benign keratinocytic lesion
    "DF",       # 4: Dermatofibroma
    "INF",      # 5: Inflammatory and infectious
    "MAL_OTH",  # 6: Other malignant proliferations
    "MEL",      # 7: Melanoma
    "NV",       # 8: Melanocytic nevus
    "SCCKA",    # 9: SCC/Keratoacanthoma
    "VASC",     # 10: Vascular lesions
]

# Mapping from various dataset labels to Tricorder-3 classes
LABEL_MAPPING = {
    # HAM10000 / ISIC standard labels
    "akiec": "AKIEC",
    "bcc": "BCC",
    "bkl": "BKL",
    "df": "DF",
    "mel": "MEL",
    "nv": "NV",
    "vasc": "VASC",
    
    # ISIC 2019 additional
    "scc": "SCCKA",
    "ak": "AKIEC",
    
    # Full names
    "melanoma": "MEL",
    "nevus": "NV",
    "basal cell carcinoma": "BCC",
    "actinic keratosis": "AKIEC",
    "benign keratosis": "BKL",
    "dermatofibroma": "DF",
    "vascular lesion": "VASC",
    "vascular": "VASC",
    "squamous cell carcinoma": "SCCKA",
    "keratoacanthoma": "SCCKA",
    
    # Other benign -> BEN_OTH
    "benign": "BEN_OTH",
    "other": "BEN_OTH",
    "unknown": "BEN_OTH",
    "unk": "BEN_OTH",
    
    # Inflammatory -> INF
    "inflammatory": "INF",
    "infection": "INF",
    "psoriasis": "INF",
    "eczema": "INF",
    
    # Other malignant -> MAL_OTH
    "malignant": "MAL_OTH",
    "merkel": "MAL_OTH",
    "kaposi": "MAL_OTH",
}

def map_label_to_tricorder(label: str) -> Optional[str]:
    """Map any label to Tricorder-3 class."""
    if not label:
        return None
    
    label_lower = str(label).lower().strip()
    
    # Direct mapping
    if label_lower in LABEL_MAPPING:
        return LABEL_MAPPING[label_lower]
    
    # Check if already a valid class
    if label_lower.upper() in TRICORDER_CLASSES:
        return label_lower.upper()
    
    # Partial matching
    for key, value in LABEL_MAPPING.items():
        if key in label_lower or label_lower in key:
            return value
    
    return None


# ============================================================================
# Dataset Downloaders
# ============================================================================

class DatasetDownloader:
    """Base class for dataset downloaders."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download(self) -> Tuple[List[str], List[str], List[Dict]]:
        """Download dataset and return (image_paths, labels, metadata)."""
        raise NotImplementedError


class HuggingFaceDownloader(DatasetDownloader):
    """Download datasets from HuggingFace."""
    
    def __init__(self, output_dir: str, dataset_id: str, name: str = None):
        super().__init__(output_dir)
        self.dataset_id = dataset_id
        self.name = name or dataset_id.split("/")[-1]
        
    def download(self) -> Tuple[List[str], List[str], List[Dict]]:
        from datasets import load_dataset
        
        print(f"\n{'='*60}")
        print(f"Downloading: {self.dataset_id}")
        print(f"{'='*60}")
        
        try:
            dataset = load_dataset(self.dataset_id, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load {self.dataset_id}: {e}")
            return [], [], []
        
        image_paths = []
        labels = []
        metadata = []
        
        # Process all splits
        for split_name in dataset.keys():
            split = dataset[split_name]
            print(f"Processing split: {split_name} ({len(split)} samples)")
            
            save_dir = self.output_dir / self.name / split_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, item in enumerate(tqdm(split, desc=f"  {split_name}")):
                try:
                    # Get image
                    if 'image' in item:
                        img = item['image']
                    elif 'img' in item:
                        img = item['img']
                    else:
                        continue
                    
                    # Get label
                    label = None
                    for key in ['label', 'dx', 'diagnosis', 'class', 'target']:
                        if key in item:
                            label = item[key]
                            break
                    
                    if label is None:
                        continue
                    
                    # Map to Tricorder class
                    if isinstance(label, int):
                        # Try to get label name from features
                        if hasattr(split.features.get('label', None), 'names'):
                            label = split.features['label'].names[label]
                    
                    tricorder_label = map_label_to_tricorder(str(label))
                    if tricorder_label is None:
                        continue
                    
                    # Save image
                    img_path = save_dir / f"{self.name}_{split_name}_{idx:06d}.jpg"
                    if isinstance(img, Image.Image):
                        img = img.convert('RGB')
                        img.save(img_path, 'JPEG', quality=95)
                    else:
                        continue
                    
                    image_paths.append(str(img_path))
                    labels.append(tricorder_label)
                    
                    # Get metadata
                    meta = {}
                    if 'age' in item:
                        meta['age'] = item['age']
                    if 'sex' in item:
                        meta['gender'] = item['sex']
                    elif 'gender' in item:
                        meta['gender'] = item['gender']
                    if 'localization' in item:
                        meta['location'] = item['localization']
                    elif 'location' in item:
                        meta['location'] = item['location']
                    
                    metadata.append(meta)
                    
                except Exception as e:
                    continue
        
        print(f"Downloaded {len(image_paths)} images from {self.dataset_id}")
        return image_paths, labels, metadata


class KaggleDownloader(DatasetDownloader):
    """Download datasets from Kaggle."""
    
    def __init__(self, output_dir: str, dataset_id: str, name: str = None):
        super().__init__(output_dir)
        self.dataset_id = dataset_id
        self.name = name or dataset_id.split("/")[-1]
        
    def download(self) -> Tuple[List[str], List[str], List[Dict]]:
        import subprocess
        
        print(f"\n{'='*60}")
        print(f"Downloading from Kaggle: {self.dataset_id}")
        print(f"{'='*60}")
        
        save_dir = self.output_dir / self.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if kaggle is configured
        kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_config.exists():
            print(f"WARNING: Kaggle not configured. Please:")
            print(f"  1. Go to https://www.kaggle.com/account")
            print(f"  2. Create New API Token")
            print(f"  3. Save to ~/.kaggle/kaggle.json")
            print(f"  4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return [], [], []
        
        try:
            # Download dataset
            cmd = f"kaggle datasets download -d {self.dataset_id} -p {save_dir} --unzip"
            subprocess.run(cmd.split(), check=True)
            
            # Find and process the data
            return self._process_downloaded_data(save_dir)
            
        except Exception as e:
            print(f"Failed to download from Kaggle: {e}")
            return [], [], []
    
    def _process_downloaded_data(self, data_dir: Path) -> Tuple[List[str], List[str], List[Dict]]:
        """Process downloaded Kaggle data."""
        image_paths = []
        labels = []
        metadata = []
        
        # Look for CSV files
        csv_files = list(data_dir.rglob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Find image and label columns
                img_col = None
                label_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if 'image' in col_lower or 'file' in col_lower:
                        img_col = col
                    if 'label' in col_lower or 'dx' in col_lower or 'class' in col_lower:
                        label_col = col
                
                if img_col is None or label_col is None:
                    continue
                
                # Find image directory
                img_dirs = [d for d in data_dir.rglob("*") if d.is_dir() and any(d.glob("*.jpg"))]
                
                for _, row in df.iterrows():
                    img_name = str(row[img_col])
                    if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_name += '.jpg'
                    
                    # Find image file
                    img_path = None
                    for img_dir in img_dirs:
                        potential_path = img_dir / img_name
                        if potential_path.exists():
                            img_path = potential_path
                            break
                    
                    if img_path is None:
                        continue
                    
                    # Map label
                    tricorder_label = map_label_to_tricorder(str(row[label_col]))
                    if tricorder_label is None:
                        continue
                    
                    image_paths.append(str(img_path))
                    labels.append(tricorder_label)
                    
                    # Metadata
                    meta = {}
                    if 'age' in df.columns:
                        meta['age'] = row['age']
                    if 'sex' in df.columns:
                        meta['gender'] = row['sex']
                    if 'localization' in df.columns:
                        meta['location'] = row['localization']
                    metadata.append(meta)
                    
            except Exception as e:
                continue
        
        print(f"Processed {len(image_paths)} images from Kaggle dataset")
        return image_paths, labels, metadata


class URLDownloader(DatasetDownloader):
    """Download datasets from direct URLs."""
    
    def __init__(self, output_dir: str, url: str, name: str):
        super().__init__(output_dir)
        self.url = url
        self.name = name
        
    def download(self) -> Tuple[List[str], List[str], List[Dict]]:
        print(f"\n{'='*60}")
        print(f"Downloading: {self.name}")
        print(f"URL: {self.url}")
        print(f"{'='*60}")
        
        save_dir = self.output_dir / self.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Download file
        filename = self.url.split("/")[-1]
        filepath = save_dir / filename
        
        if not filepath.exists():
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(self.url, filepath)
            except Exception as e:
                print(f"Failed to download: {e}")
                return [], [], []
        
        # Extract if archive
        if filename.endswith('.zip'):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as z:
                z.extractall(save_dir)
        elif filename.endswith(('.tar.gz', '.tgz')):
            print(f"Extracting {filename}...")
            with tarfile.open(filepath, 'r:gz') as t:
                t.extractall(save_dir)
        
        # Process extracted data
        return self._process_data(save_dir)
    
    def _process_data(self, data_dir: Path) -> Tuple[List[str], List[str], List[Dict]]:
        """Process downloaded data - subclasses can override."""
        return [], [], []


# ============================================================================
# Dataset Collection
# ============================================================================

HUGGINGFACE_DATASETS = [
    # Primary datasets with good class coverage
    "marmal88/skin_cancer",
    "NeuronZero/Skin-Cancer-ISIC",
    "harshildarji/isic-2020",
]

KAGGLE_DATASETS = [
    # HAM10000 - the gold standard
    ("kmader/skin-cancer-mnist-ham10000", "ham10000"),
    # ISIC datasets
    ("nodoubttome/skin-cancer9-classesisic", "isic_9class"),
]


# ============================================================================
# Main Functions
# ============================================================================

def download_all_datasets(
    output_dir: str,
    skip_kaggle: bool = False,
    skip_large: bool = False,
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Download all available datasets.
    
    Returns:
        Combined (image_paths, labels, metadata) from all datasets
    """
    all_images = []
    all_labels = []
    all_metadata = []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download from HuggingFace
    print("\n" + "="*70)
    print("DOWNLOADING FROM HUGGINGFACE")
    print("="*70)
    
    for dataset_id in HUGGINGFACE_DATASETS:
        try:
            downloader = HuggingFaceDownloader(output_dir, dataset_id)
            images, labels, metadata = downloader.download()
            all_images.extend(images)
            all_labels.extend(labels)
            all_metadata.extend(metadata)
        except Exception as e:
            print(f"Failed {dataset_id}: {e}")
    
    # Download from Kaggle
    if not skip_kaggle:
        print("\n" + "="*70)
        print("DOWNLOADING FROM KAGGLE")
        print("="*70)
        
        for dataset_id, name in KAGGLE_DATASETS:
            try:
                downloader = KaggleDownloader(output_dir, dataset_id, name)
                images, labels, metadata = downloader.download()
                all_images.extend(images)
                all_labels.extend(labels)
                all_metadata.extend(metadata)
            except Exception as e:
                print(f"Failed {dataset_id}: {e}")
    
    return all_images, all_labels, all_metadata


def create_unified_dataset(
    image_paths: List[str],
    labels: List[str],
    metadata: List[Dict],
    output_dir: str,
    copy_images: bool = False,
) -> str:
    """
    Create a unified dataset with consistent format.
    
    Args:
        image_paths: List of image file paths
        labels: List of Tricorder-3 class labels
        metadata: List of metadata dicts
        output_dir: Output directory
        copy_images: Whether to copy images to output dir
    
    Returns:
        Path to the created labels.csv
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Remove duplicates based on image hash
    print("\nRemoving duplicates...")
    unique_data = {}
    
    for img_path, label, meta in tqdm(zip(image_paths, labels, metadata), total=len(image_paths)):
        try:
            # Hash image content
            with open(img_path, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
            
            if img_hash not in unique_data:
                unique_data[img_hash] = (img_path, label, meta)
        except:
            continue
    
    print(f"Unique images: {len(unique_data)} (removed {len(image_paths) - len(unique_data)} duplicates)")
    
    # Create DataFrame
    records = []
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    for idx, (img_hash, (img_path, label, meta)) in enumerate(tqdm(unique_data.items(), desc="Creating dataset")):
        # New filename
        new_filename = f"img_{idx:06d}.jpg"
        
        if copy_images:
            # Copy image
            new_path = images_dir / new_filename
            shutil.copy2(img_path, new_path)
            img_path = str(new_path)
        
        record = {
            'image': new_filename if copy_images else img_path,
            'label': label,
            'label_idx': TRICORDER_CLASSES.index(label),
            'age': meta.get('age', ''),
            'gender': meta.get('gender', ''),
            'location': meta.get('location', ''),
            'source_path': img_path,
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Save CSV
    csv_path = output_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    
    # Print statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"\nClass distribution:")
    
    class_counts = df['label'].value_counts()
    for label in TRICORDER_CLASSES:
        count = class_counts.get(label, 0)
        pct = count / len(df) * 100 if len(df) > 0 else 0
        risk = "HIGH" if label in ["BCC", "MAL_OTH", "MEL", "SCCKA"] else \
               "MEDIUM" if label in ["AKIEC", "BKL", "VASC"] else "BENIGN"
        bar = "█" * int(pct / 2)
        print(f"  {label:8} ({risk:6}): {count:6} ({pct:5.1f}%) {bar}")
    
    # Check for missing classes
    missing = [c for c in TRICORDER_CLASSES if c not in class_counts.index]
    if missing:
        print(f"\n⚠️  WARNING: Missing classes: {missing}")
        print("   These classes have no training data!")
    
    print(f"\nDataset saved to: {csv_path}")
    
    return str(csv_path)


def create_synthetic_samples_for_missing_classes(
    output_dir: str,
    existing_df: pd.DataFrame,
    samples_per_class: int = 100,
) -> pd.DataFrame:
    """
    Create synthetic samples for classes with no data.
    Uses data augmentation on similar classes.
    """
    from torchvision import transforms
    
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Find missing classes
    existing_classes = set(existing_df['label'].unique())
    missing_classes = [c for c in TRICORDER_CLASSES if c not in existing_classes]
    
    if not missing_classes:
        print("No missing classes - all 11 classes have data!")
        return existing_df
    
    print(f"\nCreating synthetic data for missing classes: {missing_classes}")
    
    # Map missing classes to similar existing classes for augmentation
    similar_classes = {
        "BEN_OTH": ["DF", "NV", "BKL"],  # Other benign
        "INF": ["BKL", "DF"],  # Inflammatory
        "MAL_OTH": ["MEL", "BCC"],  # Other malignant
        "SCCKA": ["BCC", "AKIEC"],  # SCC similar to BCC/AKIEC
    }
    
    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    ])
    
    new_records = []
    
    for missing_class in missing_classes:
        source_classes = similar_classes.get(missing_class, ["NV"])
        
        # Get source images
        source_df = existing_df[existing_df['label'].isin(source_classes)]
        if len(source_df) == 0:
            source_df = existing_df.sample(min(100, len(existing_df)))
        
        print(f"  Creating {samples_per_class} samples for {missing_class} from {source_classes}")
        
        for i in range(samples_per_class):
            # Random source image
            source_row = source_df.sample(1).iloc[0]
            source_path = source_row['source_path'] if 'source_path' in source_row else source_row['image']
            
            try:
                img = Image.open(source_path).convert('RGB')
                
                # Apply augmentation
                img = augment(img)
                
                # Save
                new_idx = len(existing_df) + len(new_records)
                new_filename = f"synthetic_{missing_class}_{i:04d}.jpg"
                new_path = images_dir / new_filename
                img.save(new_path, 'JPEG', quality=90)
                
                new_records.append({
                    'image': str(new_path),
                    'label': missing_class,
                    'label_idx': TRICORDER_CLASSES.index(missing_class),
                    'age': source_row.get('age', ''),
                    'gender': source_row.get('gender', ''),
                    'location': source_row.get('location', ''),
                    'source_path': str(new_path),
                    'is_synthetic': True,
                })
            except Exception as e:
                continue
    
    # Combine
    new_df = pd.DataFrame(new_records)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    print(f"Added {len(new_records)} synthetic samples")
    
    return combined_df


def balance_dataset(
    df: pd.DataFrame,
    max_samples_per_class: int = 2000,
    min_samples_per_class: int = 500,
) -> pd.DataFrame:
    """
    Balance dataset by oversampling minority classes and undersampling majority.
    """
    print("\nBalancing dataset...")
    
    balanced_dfs = []
    
    for label in TRICORDER_CLASSES:
        class_df = df[df['label'] == label]
        n_samples = len(class_df)
        
        if n_samples == 0:
            continue
        
        if n_samples > max_samples_per_class:
            # Undersample
            class_df = class_df.sample(max_samples_per_class, random_state=42)
        elif n_samples < min_samples_per_class:
            # Oversample
            n_needed = min_samples_per_class - n_samples
            oversampled = class_df.sample(n_needed, replace=True, random_state=42)
            class_df = pd.concat([class_df, oversampled], ignore_index=True)
        
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"Balanced dataset: {len(balanced_df)} samples")
    
    return balanced_df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for Tricorder-3 competition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="training_data",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--skip-kaggle", action="store_true",
        help="Skip Kaggle downloads (if not configured)"
    )
    parser.add_argument(
        "--skip-large", action="store_true",
        help="Skip large datasets (>1GB)"
    )
    parser.add_argument(
        "--copy-images", action="store_true",
        help="Copy images to output directory (uses more disk space)"
    )
    parser.add_argument(
        "--create-synthetic", action="store_true", default=True,
        help="Create synthetic samples for missing classes"
    )
    parser.add_argument(
        "--balance", action="store_true", default=True,
        help="Balance the dataset"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=2000,
        help="Maximum samples per class after balancing"
    )
    parser.add_argument(
        "--min-per-class", type=int, default=500,
        help="Minimum samples per class (oversample if needed)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("TRICORDER-3 DATASET PREPARATION")
    print("="*70)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Target classes: {TRICORDER_CLASSES}")
    
    # Step 1: Download datasets
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING DATASETS")
    print("="*70)
    
    images, labels, metadata = download_all_datasets(
        args.output_dir,
        skip_kaggle=args.skip_kaggle,
        skip_large=args.skip_large,
    )
    
    if len(images) == 0:
        print("\n❌ No images downloaded. Please check your internet connection and try again.")
        print("\nAlternative: Download manually from:")
        print("  - Kaggle HAM10000: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000")
        print("  - ISIC Archive: https://www.isic-archive.com/")
        return
    
    # Step 2: Create unified dataset
    print("\n" + "="*70)
    print("STEP 2: CREATING UNIFIED DATASET")
    print("="*70)
    
    csv_path = create_unified_dataset(
        images, labels, metadata,
        args.output_dir,
        copy_images=args.copy_images,
    )
    
    # Load the created dataset
    df = pd.read_csv(csv_path)
    
    # Step 3: Handle missing classes
    if args.create_synthetic:
        print("\n" + "="*70)
        print("STEP 3: HANDLING MISSING CLASSES")
        print("="*70)
        
        df = create_synthetic_samples_for_missing_classes(
            args.output_dir,
            df,
            samples_per_class=args.min_per_class,
        )
    
    # Step 4: Balance dataset
    if args.balance:
        print("\n" + "="*70)
        print("STEP 4: BALANCING DATASET")
        print("="*70)
        
        df = balance_dataset(
            df,
            max_samples_per_class=args.max_per_class,
            min_samples_per_class=args.min_per_class,
        )
    
    # Save final dataset
    final_csv = Path(args.output_dir) / "labels_final.csv"
    df.to_csv(final_csv, index=False)
    
    print("\n" + "="*70)
    print("FINAL DATASET READY")
    print("="*70)
    
    print(f"\nDataset saved to: {final_csv}")
    print(f"Total samples: {len(df)}")
    
    print("\nFinal class distribution:")
    for label in TRICORDER_CLASSES:
        count = len(df[df['label'] == label])
        print(f"  {label:8}: {count:5}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"""
To train your model, run:

    python custom_model/train.py \\
        --data-dir {args.output_dir} \\
        --labels-file {final_csv} \\
        --epochs 50 \\
        --batch-size 16 \\
        --model-type balanced

Or for more accuracy (longer training):

    python custom_model/train.py \\
        --data-dir {args.output_dir} \\
        --labels-file {final_csv} \\
        --epochs 100 \\
        --batch-size 32 \\
        --model-type balanced \\
        --learning-rate 5e-4
""")


if __name__ == "__main__":
    main()
