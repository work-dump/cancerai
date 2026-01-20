#!/usr/bin/env python3
"""
Dataset Download Script for Tricorder-3 Competition

Downloads skin lesion datasets from multiple sources:
- Kaggle (HAM10000, ISIC 2019, ISIC 2020, etc.)
- HuggingFace (skin cancer datasets)
- Direct URLs (PH2, BCN20000, etc.)
- ISIC Archive

Total potential data: 500,000+ images

Usage:
    python custom_model/download_datasets.py              # Download all enabled
    python custom_model/download_datasets.py --all        # Download everything
    python custom_model/download_datasets.py --source kaggle  # Kaggle only
    python custom_model/download_datasets.py --list       # List all datasets
"""

import os
import sys
import argparse
import shutil
import zipfile
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# Tricorder-3 Class Mapping
# ============================================================================

TRICORDER_CLASSES = [
    "AKIEC",    # 0: Actinic keratosis
    "BCC",      # 1: Basal cell carcinoma
    "BEN_OTH",  # 2: Other benign
    "BKL",      # 3: Benign keratosis
    "DF",       # 4: Dermatofibroma
    "INF",      # 5: Inflammatory
    "MAL_OTH",  # 6: Other malignant
    "MEL",      # 7: Melanoma
    "NV",       # 8: Nevus
    "SCCKA",    # 9: SCC/Keratoacanthoma
    "VASC",     # 10: Vascular
]

LABEL_MAPPING = {
    # HAM10000 / ISIC standard
    "akiec": "AKIEC", "bcc": "BCC", "bkl": "BKL", "df": "DF",
    "mel": "MEL", "nv": "NV", "vasc": "VASC",
    
    # ISIC 2019
    "ak": "AKIEC", "scc": "SCCKA", "melanoma": "MEL", "nevus": "NV",
    
    # Full names
    "squamous cell carcinoma": "SCCKA",
    "basal cell carcinoma": "BCC",
    "actinic keratosis": "AKIEC",
    "benign keratosis": "BKL",
    "dermatofibroma": "DF",
    "vascular lesion": "VASC",
    "melanocytic nevus": "NV",
    "seborrheic keratosis": "BKL",
    "lichenoid keratosis": "BKL",
    "solar lentigo": "BKL",
    "lentigo nos": "BKL",
    "cafe-au-lait macule": "BEN_OTH",
    "atypical melanocytic proliferation": "MEL",
    "pigmented benign keratosis": "BKL",
    
    # BCN20000 classes
    "melanocytic nevi": "NV",
    "melanoma": "MEL",
    "benign keratosis": "BKL",
    "basal cell carcinoma": "BCC",
    "actinic keratosis": "AKIEC",
    "vascular lesion": "VASC",
    "dermatofibroma": "DF",
    "squamous cell carcinoma": "SCCKA",
    
    # Generic mappings
    "benign": "BEN_OTH",
    "malignant": "MAL_OTH",
    "unknown": "BEN_OTH",
    "other": "BEN_OTH",
}


def map_label(label: str) -> Optional[str]:
    """Map dataset label to Tricorder-3 class."""
    if not label:
        return None
    
    label_lower = str(label).lower().strip()
    
    if label_lower in LABEL_MAPPING:
        return LABEL_MAPPING[label_lower]
    
    if label_lower.upper() in TRICORDER_CLASSES:
        return label_lower.upper()
    
    for key, value in LABEL_MAPPING.items():
        if key in label_lower or label_lower in key:
            return value
    
    return None


# ============================================================================
# Dataset Definitions
# ============================================================================

KAGGLE_DATASETS = {
    "ham10000": {
        "id": "kmader/skin-cancer-mnist-ham10000",
        "name": "HAM10000",
        "description": "10,015 dermoscopic images, 7 classes",
        "size": "~3 GB",
    },
    "isic_2019": {
        "id": "andrewmvd/isic-2019",
        "name": "ISIC 2019",
        "description": "25,331 images, 8 diagnostic categories",
        "size": "~9 GB",
    },
    "isic_2020": {
        "id": "cdeotte/jpeg-melanoma-256x256",
        "name": "ISIC 2020 (256x256)",
        "description": "33,126 images for melanoma detection",
        "size": "~2 GB",
    },
    "skin_cancer_9class": {
        "id": "nodoubttome/skin-cancer9-classesisic",
        "name": "Skin Cancer 9 Classes",
        "description": "9-class ISIC skin cancer dataset",
        "size": "~4 GB",
    },
    "dermnet": {
        "id": "shubhamgoel27/dermnet",
        "name": "DermNet",
        "description": "23,000 images, 23 disease classes",
        "size": "~5 GB",
    },
    "isic_2018": {
        "id": "kmader/skin-cancer-isic",
        "name": "ISIC 2018",
        "description": "ISIC 2018 challenge dataset",
        "size": "~2 GB",
    },
    "skin_lesion_7class": {
        "id": "surajghuwalewala/ham1000-segmentation-and-classification",
        "name": "HAM10000 Segmentation",
        "description": "HAM10000 with segmentation masks",
        "size": "~1 GB",
    },
}

HUGGINGFACE_DATASETS = {
    "marmal88_skin": {
        "id": "marmal88/skin_cancer",
        "name": "Skin Cancer (HF)",
        "description": "Skin cancer classification dataset",
    },
    "isic_hf": {
        "id": "NeuronZero/Skin-Cancer-ISIC",
        "name": "ISIC (HuggingFace)",
        "description": "ISIC dataset on HuggingFace",
    },
}

DIRECT_DOWNLOADS = {
    "ph2": {
        "url": "https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar?dl=1",
        "name": "PH2 Dataset",
        "description": "200 dermoscopic images (40 melanoma, 80 atypical nevi, 80 common nevi)",
        "size": "~425 MB",
    },
}


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "train_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        return {}
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {config_path}")
        return config or {}
    except ImportError:
        return {}
    except Exception:
        return {}


def get_config_value(config: dict, *keys, default=None):
    """Get nested config value safely."""
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value if value is not None else default


# ============================================================================
# Download Functions
# ============================================================================

def check_kaggle_setup() -> bool:
    """Check if Kaggle API is properly configured."""
    try:
        import kaggle
        kaggle.api.authenticate()
        print("✓ Kaggle API authenticated")
        return True
    except ImportError:
        print("✗ Kaggle not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"✗ Kaggle auth failed: {e}")
        print("\nSetup Kaggle API:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token'")
        print("  3. Save kaggle.json to:")
        print("     Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print("     Linux:   ~/.kaggle/kaggle.json")
        return False


def download_kaggle_dataset(dataset_id: str, output_dir: Path) -> bool:
    """Download a Kaggle dataset."""
    try:
        import kaggle
        
        print(f"\n📥 Downloading from Kaggle: {dataset_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        kaggle.api.dataset_download_files(
            dataset_id,
            path=str(output_dir),
            unzip=True,
            quiet=False,
        )
        
        print(f"✓ Downloaded: {dataset_id}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def download_huggingface_dataset(dataset_id: str, output_dir: Path) -> bool:
    """Download a HuggingFace dataset."""
    try:
        from datasets import load_dataset
        from PIL import Image
        
        print(f"\n📥 Downloading from HuggingFace: {dataset_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = load_dataset(dataset_id, trust_remote_code=True)
        
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        records = []
        idx = 0
        
        for split_name in dataset.keys():
            split = dataset[split_name]
            print(f"  Processing {split_name}: {len(split)} samples")
            
            for item in tqdm(split, desc=f"    {split_name}"):
                try:
                    # Get image
                    img = item.get('image') or item.get('img')
                    if img is None:
                        continue
                    
                    # Get label
                    label = None
                    for key in ['label', 'dx', 'diagnosis', 'class', 'target']:
                        if key in item:
                            label = item[key]
                            break
                    
                    if label is None:
                        continue
                    
                    # Map label
                    if isinstance(label, int) and hasattr(split.features.get('label', None), 'names'):
                        label = split.features['label'].names[label]
                    
                    mapped_label = map_label(str(label))
                    if mapped_label is None:
                        continue
                    
                    # Save image
                    img_path = images_dir / f"hf_{idx:06d}.jpg"
                    if isinstance(img, Image.Image):
                        img.convert('RGB').save(img_path, 'JPEG', quality=95)
                    
                    records.append({
                        "image": str(img_path),
                        "label": mapped_label,
                    })
                    idx += 1
                    
                except Exception:
                    continue
        
        # Save metadata
        if records:
            df = pd.DataFrame(records)
            df.to_csv(output_dir / "labels.csv", index=False)
        
        print(f"✓ Downloaded {len(records)} images")
        return True
        
    except ImportError:
        print("✗ HuggingFace datasets not installed. Run: pip install datasets")
        return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def download_url(url: str, output_path: Path) -> bool:
    """Download file from URL with progress bar."""
    try:
        print(f"\n📥 Downloading: {url}")
        
        # Get file size
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="  Downloading") as pbar:
            def reporthook(count, block_size, total_size):
                pbar.update(block_size)
            
            urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
        
        print(f"✓ Downloaded to: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract zip, tar, or rar archive."""
    try:
        print(f"  Extracting: {archive_path.name}")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extractall(output_dir)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as t:
                t.extractall(output_dir)
        elif archive_path.suffix == '.rar':
            try:
                import rarfile
                with rarfile.RarFile(archive_path, 'r') as r:
                    r.extractall(output_dir)
            except ImportError:
                print("    ⚠ Install rarfile: pip install rarfile")
                return False
        else:
            print(f"    ⚠ Unknown archive format: {archive_path.suffix}")
            return False
        
        print(f"  ✓ Extracted")
        return True
        
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


# ============================================================================
# Dataset Processing Functions
# ============================================================================

def process_ham10000(data_dir: Path) -> Tuple[List[str], List[str], List[dict]]:
    """Process HAM10000 dataset."""
    print("\n📊 Processing HAM10000...")
    
    # Find metadata CSV
    csv_path = None
    for pattern in ["*metadata*.csv", "*HAM*.csv", "*.csv"]:
        matches = list(data_dir.rglob(pattern))
        if matches:
            csv_path = matches[0]
            break
    
    if csv_path is None:
        print("  ✗ No metadata CSV found")
        return [], [], []
    
    df = pd.read_csv(csv_path)
    print(f"  Found {len(df)} entries in {csv_path.name}")
    
    # Find image directories
    image_dirs = list(data_dir.rglob("*images*")) + [data_dir]
    
    images, labels, metadata = [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
        img_id = str(row.get("image_id", row.iloc[0]))
        label = map_label(str(row.get("dx", row.get("label", ""))))
        
        if label is None:
            continue
        
        # Find image
        img_path = None
        for img_dir in image_dirs:
            for ext in [".jpg", ".jpeg", ".png", ""]:
                for candidate in img_dir.glob(f"*{img_id}*{ext}"):
                    if candidate.is_file() and candidate.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        img_path = candidate
                        break
                if img_path:
                    break
            if img_path:
                break
        
        if img_path is None:
            continue
        
        images.append(str(img_path))
        labels.append(label)
        metadata.append({
            "age": row.get("age"),
            "sex": row.get("sex"),
            "localization": row.get("localization"),
        })
    
    print(f"  ✓ Processed {len(images)} images")
    return images, labels, metadata


def process_isic_2019(data_dir: Path) -> Tuple[List[str], List[str], List[dict]]:
    """Process ISIC 2019 dataset."""
    print("\n📊 Processing ISIC 2019...")
    
    # Find ground truth
    csv_path = None
    for pattern in ["*GroundTruth*.csv", "*Train*.csv", "*.csv"]:
        matches = list(data_dir.rglob(pattern))
        if matches:
            csv_path = matches[0]
            break
    
    if csv_path is None:
        return process_folder_dataset(data_dir)
    
    df = pd.read_csv(csv_path)
    
    # ISIC 2019 uses one-hot encoding
    class_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    available_cols = [c for c in class_cols if c in df.columns]
    
    # Find images
    img_dirs = list(data_dir.rglob("*Input*")) + list(data_dir.rglob("*image*")) + [data_dir]
    
    images, labels, metadata = [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
        img_name = str(row.get("image", row.iloc[0]))
        
        # Get label from one-hot
        label = None
        for col in available_cols:
            if row.get(col, 0) == 1:
                label = map_label(col)
                break
        
        if label is None:
            continue
        
        # Find image
        img_path = None
        for img_dir in img_dirs:
            for ext in ["", ".jpg", ".jpeg", ".png"]:
                candidate = img_dir / f"{img_name}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path:
                break
        
        if img_path is None:
            continue
        
        images.append(str(img_path))
        labels.append(label)
        metadata.append({})
    
    print(f"  ✓ Processed {len(images)} images")
    return images, labels, metadata


def process_folder_dataset(data_dir: Path) -> Tuple[List[str], List[str], List[dict]]:
    """Process dataset with folder-based class structure."""
    print(f"\n📊 Processing folder dataset: {data_dir.name}")
    
    images, labels, metadata = [], [], []
    
    # Look for class folders
    for subdir in data_dir.rglob("*"):
        if not subdir.is_dir():
            continue
        
        label = map_label(subdir.name)
        if label is None:
            continue
        
        for img_path in subdir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                images.append(str(img_path))
                labels.append(label)
                metadata.append({})
    
    print(f"  ✓ Processed {len(images)} images")
    return images, labels, metadata


def process_generic_dataset(data_dir: Path) -> Tuple[List[str], List[str], List[dict]]:
    """Process any dataset by finding CSV or folder structure."""
    # Try CSV first
    csv_files = list(data_dir.rglob("*.csv"))
    
    all_images, all_labels, all_metadata = [], [], []
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            
            # Find columns
            img_col = None
            label_col = None
            
            for col in df.columns:
                col_l = col.lower()
                if any(x in col_l for x in ["image", "file", "name", "id", "path"]):
                    img_col = col
                if any(x in col_l for x in ["label", "class", "dx", "diagnosis", "target", "category"]):
                    label_col = col
            
            if img_col is None or label_col is None:
                continue
            
            for _, row in df.iterrows():
                img_ref = str(row[img_col])
                label = map_label(str(row[label_col]))
                
                if label is None:
                    continue
                
                # Find image file
                img_path = None
                for candidate in data_dir.rglob(f"*{img_ref}*"):
                    if candidate.is_file() and candidate.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        img_path = candidate
                        break
                
                if img_path:
                    all_images.append(str(img_path))
                    all_labels.append(label)
                    all_metadata.append({})
                    
        except Exception:
            continue
    
    # If no CSV worked, try folder structure
    if not all_images:
        return process_folder_dataset(data_dir)
    
    return all_images, all_labels, all_metadata


# ============================================================================
# Dataset Balancing
# ============================================================================

def balance_dataset(
    images: List[str],
    labels: List[str],
    metadata: List[dict],
    max_per_class: int = 3000,
    min_per_class: int = 500,
) -> Tuple[List[str], List[str], List[dict]]:
    """Balance dataset by under/oversampling."""
    print(f"\n⚖️ Balancing dataset (min={min_per_class}, max={max_per_class})...")
    
    # Group by class
    class_data = defaultdict(list)
    for img, lbl, meta in zip(images, labels, metadata):
        class_data[lbl].append((img, lbl, meta))
    
    balanced = []
    
    for label in TRICORDER_CLASSES:
        samples = class_data.get(label, [])
        n = len(samples)
        
        if n == 0:
            print(f"  ⚠ {label:8}: 0 samples")
            continue
        
        if n > max_per_class:
            np.random.shuffle(samples)
            samples = samples[:max_per_class]
            print(f"  {label:8}: {n:5} → {max_per_class} (undersampled)")
        elif n < min_per_class:
            additional = min_per_class - n
            oversampled = [samples[i % n] for i in range(additional)]
            samples = samples + oversampled
            print(f"  {label:8}: {n:5} → {len(samples)} (oversampled)")
        else:
            print(f"  {label:8}: {n:5}")
        
        balanced.extend(samples)
    
    np.random.shuffle(balanced)
    
    if balanced:
        images, labels, metadata = zip(*balanced)
        return list(images), list(labels), list(metadata)
    return [], [], []


# ============================================================================
# Main
# ============================================================================

def create_labels_csv(images, labels, metadata, output_path):
    """Create final labels CSV."""
    records = []
    for img, lbl, meta in zip(images, labels, metadata):
        records.append({
            "image": img,
            "label": lbl,
            "label_idx": TRICORDER_CLASSES.index(lbl),
            "age": meta.get("age", ""),
            "sex": meta.get("sex", ""),
            "localization": meta.get("localization", ""),
        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Total: {len(df)} images")
    print("\n  Class distribution:")
    for label in TRICORDER_CLASSES:
        count = len(df[df["label"] == label])
        pct = count / len(df) * 100 if len(df) > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {label:8}: {count:5} ({pct:5.1f}%) {bar}")


def main():
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description="Download ALL available datasets for Tricorder-3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--output-dir", type=str,
        default=get_config_value(config, "download", "output_dir", default="training_data"))
    parser.add_argument("--source", choices=["kaggle", "huggingface", "direct", "all"],
        default="all", help="Download from specific source")
    parser.add_argument("--dataset", type=str, help="Download specific dataset by key")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    parser.add_argument("--max-per-class", type=int,
        default=get_config_value(config, "download", "max_per_class", default=3000))
    parser.add_argument("--min-per-class", type=int,
        default=get_config_value(config, "download", "min_per_class", default=500))
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--all", action="store_true", help="Download ALL datasets")
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        print("\n" + "="*70)
        print("AVAILABLE DATASETS FOR TRICORDER-3")
        print("="*70)
        
        print("\n📦 KAGGLE DATASETS:")
        for key, info in KAGGLE_DATASETS.items():
            print(f"  {key:20} - {info['name']:25} ({info.get('size', 'N/A')})")
            print(f"  {'':20}   {info['description']}")
        
        print("\n📦 HUGGINGFACE DATASETS:")
        for key, info in HUGGINGFACE_DATASETS.items():
            print(f"  {key:20} - {info['name']}")
            print(f"  {'':20}   {info['description']}")
        
        print("\n📦 DIRECT DOWNLOAD:")
        for key, info in DIRECT_DOWNLOADS.items():
            print(f"  {key:20} - {info['name']:25} ({info.get('size', 'N/A')})")
            print(f"  {'':20}   {info['description']}")
        
        print("\n" + "="*70)
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("TRICORDER-3 DATASET DOWNLOAD")
    print("="*70)
    print(f"Output: {output_dir}")
    print(f"Source: {args.source}")
    
    all_images, all_labels, all_metadata = [], [], []
    
    # Download Kaggle datasets
    if args.source in ["kaggle", "all"]:
        if check_kaggle_setup():
            kaggle_config = get_config_value(config, "download", "kaggle", default={})
            
            for key, info in KAGGLE_DATASETS.items():
                # Skip if not enabled in config (unless --all flag)
                if not args.all and not kaggle_config.get(key, False):
                    continue
                
                if args.dataset and args.dataset != key:
                    continue
                
                dataset_dir = output_dir / f"kaggle_{key}"
                
                if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
                    download_kaggle_dataset(info["id"], dataset_dir)
                else:
                    print(f"\n⏭ {key} already exists, skipping download")
                
                # Process
                if "ham10000" in key.lower() or "ham" in key.lower():
                    imgs, lbls, meta = process_ham10000(dataset_dir)
                elif "2019" in key:
                    imgs, lbls, meta = process_isic_2019(dataset_dir)
                else:
                    imgs, lbls, meta = process_generic_dataset(dataset_dir)
                
                all_images.extend(imgs)
                all_labels.extend(lbls)
                all_metadata.extend(meta)
    
    # Download HuggingFace datasets
    if args.source in ["huggingface", "all"]:
        for key, info in HUGGINGFACE_DATASETS.items():
            if args.dataset and args.dataset != key:
                continue
            
            if not args.all:
                continue  # Skip HF by default unless --all
            
            dataset_dir = output_dir / f"hf_{key}"
            
            if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
                download_huggingface_dataset(info["id"], dataset_dir)
            
            imgs, lbls, meta = process_generic_dataset(dataset_dir)
            all_images.extend(imgs)
            all_labels.extend(lbls)
            all_metadata.extend(meta)
    
    # Download direct URLs
    if args.source in ["direct", "all"]:
        for key, info in DIRECT_DOWNLOADS.items():
            if args.dataset and args.dataset != key:
                continue
            
            if not args.all:
                continue  # Skip direct by default unless --all
            
            dataset_dir = output_dir / f"direct_{key}"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Download and extract
            archive_name = info["url"].split("/")[-1].split("?")[0]
            archive_path = dataset_dir / archive_name
            
            if not archive_path.exists():
                download_url(info["url"], archive_path)
                extract_archive(archive_path, dataset_dir)
            
            imgs, lbls, meta = process_folder_dataset(dataset_dir)
            all_images.extend(imgs)
            all_labels.extend(lbls)
            all_metadata.extend(meta)
    
    if not all_images:
        print("\n✗ No images found!")
        print("  Try: python download_datasets.py --all")
        return
    
    print(f"\n{'='*70}")
    print(f"Total collected: {len(all_images)} images")
    
    # Remove duplicates
    unique = {}
    for img, lbl, meta in zip(all_images, all_labels, all_metadata):
        if img not in unique:
            unique[img] = (lbl, meta)
    
    all_images = list(unique.keys())
    all_labels = [unique[i][0] for i in all_images]
    all_metadata = [unique[i][1] for i in all_images]
    
    print(f"After deduplication: {len(all_images)} images")
    
    # Balance
    if not args.no_balance:
        all_images, all_labels, all_metadata = balance_dataset(
            all_images, all_labels, all_metadata,
            args.max_per_class, args.min_per_class,
        )
    
    # Save
    labels_path = output_dir / "labels_final.csv"
    create_labels_csv(all_images, all_labels, all_metadata, labels_path)
    
    print(f"\n{'='*70}")
    print("✓ DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"""
Next steps:

  1. Train your model:
     python custom_model/train.py

  2. Or with custom settings:
     python custom_model/train.py --epochs 100 --batch-size 32
""")


if __name__ == "__main__":
    main()
