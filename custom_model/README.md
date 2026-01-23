# Tricorder-3 Model Training Pipeline

A competition-optimized training pipeline for the Bittensor Subnet 76 (SafeScan) Tricorder-3 skin lesion classification challenge.

## Overview

This pipeline trains a multimodal deep learning model that:
- Classifies skin lesion images into 11 diagnostic categories
- Incorporates patient demographics (age, gender, location)
- Optimizes for the competition's weighted scoring system
- Exports to ONNX format for submission

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_train.txt

# 2. Download datasets
python custom_model/download_datasets.py --source isic

# 3. Split into train/val/test with balancing
python custom_model/split_dataset.py --input training_data --output training_data_split --balance

# 4. Train
python custom_model/train.py --data-dir training_data_split

# 5. Evaluate on held-out test set
python custom_model/evaluate.py --model checkpoints/model.onnx --data-dir training_data_split/test

# 6. Submit (model at: checkpoints/model.onnx)
```

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation) ← **Start Here**
3. [Configuration](#configuration)
4. [Training](#training)
5. [Model Architecture](#model-architecture)
6. [Competition Scoring](#competition-scoring)
7. [Evaluation](#evaluation)
8. [Submission](#submission)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ GPU memory

### Step 1: Install PyTorch with CUDA

```bash
# For CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only (slower)
pip install torch torchvision
```

### Step 2: Install Training Dependencies

```bash
pip install -r requirements_train.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Dataset Preparation

> **Important**: You must prepare your dataset before training!

### Step 1: Setup Kaggle API

#### Windows

1. Create Kaggle account at https://www.kaggle.com
2. Go to **Settings** → **API** → **Create New Token**
3. Download `kaggle.json`
4. Create folder and move file:

```cmd
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

5. Install Kaggle package:

```cmd
pip install kaggle
```

#### Linux / Mac

```bash
# Create directory
mkdir -p ~/.kaggle

# Move downloaded kaggle.json
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json

# Install package
pip install kaggle
```

### Step 2: Download MILK10k (RECOMMENDED - Has All 11 Classes!)

**MILK10k is the ONLY dataset with all 11 Tricorder-3 classes!**

1. **Go to ISIC Challenge website:**
   https://challenge.isic-archive.com/landing/milk10k/

2. **Create a free ISIC account** (if you don't have one)

3. **Click "Download Data"** and download:
   - MILK10k Training Images (~8 GB)
   - MILK10k Training Metadata (CSV)
   - MILK10k Supplemental Data (optional)

4. **Extract to `training_data/isic_milk10k/`:**
   ```
   training_data/
   └── isic_milk10k/
       ├── images/
       │   ├── ISIC_xxxxxxx.jpg
       │   └── ...
       └── metadata.csv
   ```

5. **Process the dataset:**
   ```bash
   python custom_model/download_datasets.py --source isic
   ```

### Step 3: Split Dataset (RECOMMENDED for Reproducible Training)

After downloading, split your data into train/val/test sets. This ensures:
- **Reproducible results** - Same split across training sessions
- **Proper evaluation** - Test set is never seen during training
- **Resume training** - Continue from checkpoint without data leakage

```bash
# Split into 80% train, 10% val, 10% test (no balancing)
python custom_model/split_dataset.py --input training_data --output training_data_split

# Split WITH balancing (recommended for imbalanced datasets)
python custom_model/split_dataset.py --input training_data --output training_data_split --balance

# Custom balancing limits
python custom_model/split_dataset.py --input training_data --output training_data_split \
    --balance --min-per-class 500 --max-per-class 3000
```

**Note**: Balancing only affects the training set. Validation and test sets remain unbalanced for fair evaluation.

This creates:
```
training_data_split/
├── train/
│   ├── labels.csv
│   └── images/ (symlinks)
├── val/
│   ├── labels.csv
│   └── images/
└── test/
    ├── labels.csv
    └── images/
```

**Then train with the split dataset:**
```bash
python custom_model/train.py --data-dir training_data_split
```

The training code automatically detects pre-split folders and uses them without re-splitting.

### Step 4: (Optional) Add More Data from Kaggle

Edit `train_config.yaml` to enable additional datasets:

```yaml
download:
  output_dir: "training_data"
  
  # MILK10k - already downloaded manually
  isic:
    milk10k: true
  
  # Kaggle - supplement with more data for common classes
  kaggle:
    ham10000: true          # 10,015 images, 7 classes
    isic_2019: true         # 25,331 images, 8 classes
    isic_2020: false        # Melanoma only
    dermnet: false          # Large, 23 classes
  
  balance_dataset: true
  max_per_class: 3000
  min_per_class: 500
```

Then download:

```bash
python custom_model/download_datasets.py --source kaggle
```

### Step 5: Verify Dataset

```bash
# List all available datasets
python custom_model/download_datasets.py --list

# Check what was downloaded
ls -la training_data/
```

### Available Datasets

**Total potential: 100,000+ images from multiple sources**

#### ⭐ ISIC MILK10k - THE RECOMMENDED DATASET ⭐

| Key | Dataset | Images | Description |
|-----|---------|--------|-------------|
| `milk10k` | **MILK10k** | 10,480 | **ALL 11 CLASSES!** Including BEN_OTH, INF, MAL_OTH |

**This is the ONLY dataset with all 11 Tricorder-3 classes!**
- Download from: https://challenge.isic-archive.com/landing/milk10k/
- 5,240 lesions (clinical + dermoscopic pairs)
- Includes metadata: age, sex, skin tone, anatomical site
- License: CC-BY-NC

#### Kaggle Datasets (to supplement MILK10k)

| Key | Dataset | Images | Classes | Description |
|-----|---------|--------|---------|-------------|
| `ham10000` | HAM10000 | 10,015 | 7 | Primary dataset |
| `isic_2019` | ISIC 2019 | 25,331 | 8 | ISIC Challenge 2019 |
| `isic_2020` | ISIC 2020 | 33,126 | 2 | Melanoma detection |
| `dermnet` | DermNet | ~23,000 | 23 | Includes inflammatory |

#### HuggingFace & Direct Downloads

| Key | Dataset | Description |
|-----|---------|-------------|
| `marmal88_skin` | Skin Cancer (HF) | Classification dataset |
| `ph2` | PH2 | 200 images, melanoma/nevi |

### Download Commands

```bash
# ⭐ RECOMMENDED: Start with ISIC MILK10k
python custom_model/download_datasets.py --source isic

# Then add Kaggle data for more samples
python custom_model/download_datasets.py --source kaggle

# Download ALL available datasets
python custom_model/download_datasets.py --all

# List all available datasets
python custom_model/download_datasets.py --list
```

### Option B: Manual Dataset Setup

If you prefer manual download:

1. Download from Kaggle:
   - [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
   - [ISIC 2019](https://www.kaggle.com/datasets/andrewmvd/isic-2019)

2. Create a CSV file with columns:

| Column | Required | Description |
|--------|----------|-------------|
| `image` | Yes | Image path or filename |
| `label` | Yes | Class name (AKIEC, BCC, MEL, etc.) |
| `age` | No | Patient age |
| `sex` | No | male/female |
| `localization` | No | Body location |

3. Update `train_config.yaml`:

```yaml
data:
  data_dir: "path/to/images"
  labels_file: "path/to/labels.csv"
```

### Class Labels (11 Categories)

| Index | Code | Full Name | Risk Level |
|-------|------|-----------|------------|
| 0 | AKIEC | Actinic Keratosis | MEDIUM |
| 1 | BCC | Basal Cell Carcinoma | **HIGH** |
| 2 | BEN_OTH | Other Benign | BENIGN |
| 3 | BKL | Benign Keratosis | MEDIUM |
| 4 | DF | Dermatofibroma | BENIGN |
| 5 | INF | Inflammatory | BENIGN |
| 6 | MAL_OTH | Other Malignant | **HIGH** |
| 7 | MEL | Melanoma | **HIGH** |
| 8 | NV | Melanocytic Nevus | BENIGN |
| 9 | SCCKA | Squamous Cell Carcinoma | **HIGH** |
| 10 | VASC | Vascular Lesion | MEDIUM |

---

## Configuration

All settings are in `custom_model/train_config.yaml`:

```yaml
# Dataset Download Settings
download:
  output_dir: "training_data"
  
  # ISIC MILK10k - RECOMMENDED (has all 11 classes!)
  isic:
    milk10k: true
  
  # Kaggle datasets (supplement MILK10k)
  kaggle:
    ham10000: true
    isic_2019: true
  
  balance_dataset: true
  max_per_class: 3000
  min_per_class: 500

# Training Data Settings
data:
  data_dir: "training_data"
  labels_file: "training_data/labels_final.csv"
  val_split: 0.2

# Model Settings
model:
  type: "balanced"  # lightweight, balanced, or larger

# Training Settings
training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  patience: 10
  num_workers: 4
  seed: 42

# Data Augmentation
augmentation:
  use_mixup: true
  mixup_alpha: 0.4
  label_smoothing: 0.1
  weighted_sampling: true

# Checkpointing
checkpoints:
  save_dir: "checkpoints"
  save_every: 5
  export_onnx: true
  resume: null  # Set to checkpoint path to resume

# Logging
logging:
  tensorboard: true
  tensorboard_dir: "runs"
```

---

## Training

### Basic Training

```bash
python custom_model/train.py
```

### Override Settings via Command Line

```bash
python custom_model/train.py --epochs 100 --batch-size 32 --learning-rate 0.0005
```

### Resume Training

Edit `train_config.yaml`:
```yaml
checkpoints:
  resume: "checkpoints/latest.pt"
```

Or via command line:
```bash
python custom_model/train.py --resume checkpoints/latest.pt
```

### Monitor Training with TensorBoard

```bash
# In a separate terminal
tensorboard --logdir runs

# Open http://localhost:6006
```

### Training Output

During training you'll see:

```
Epoch 1/50 | Loss: 2.3456 | Val Acc: 0.4521 | Val wF1: 0.3812 | Score: 0.4015
Epoch 2/50 | Loss: 1.8234 | Val Acc: 0.5234 | Val wF1: 0.4567 | Score: 0.4721
  -> New best score: 0.4721
  -> Saved best checkpoint: checkpoints/best_model_epoch2_score_0.4721.pt
```

---

## Model Architecture

### Input

| Input | Shape | Description |
|-------|-------|-------------|
| Image | (B, 3, 512, 512) | RGB image, normalized [0,1] |
| Demographics | (B, 3) | [age, gender, location] |

### Architecture

```
EfficientNet-B0 (pretrained, frozen)
       ↓
Cross-Attention Fusion ← Demographics Encoder
       ↓
Classifier Head (MLP)
       ↓
Softmax (11 classes)
```

### Output
- 11 probabilities summing to 1.0

### Model Variants

| Type | Size | Parameters | Use Case |
|------|------|------------|----------|
| `lightweight` | ~20 MB | ~5M | Fast inference |
| `balanced` | ~45 MB | ~12M | **Recommended** |
| `larger` | ~80 MB | ~25M | Maximum accuracy |

---

## Competition Scoring

### Score Formula

```
Final Score = 0.9 × Prediction Score + 0.1 × Efficiency Score

Prediction Score = 0.5 × Accuracy + 0.5 × Weighted F1

Weighted F1 = (3×F1_HIGH + 2×F1_MEDIUM + 1×F1_BENIGN) / 6
```

### Risk Categories

| Category | Classes | Weight | Impact on Score |
|----------|---------|--------|-----------------|
| **HIGH_RISK** | BCC, MAL_OTH, MEL, SCCKA | 3× | 50% of Weighted F1 |
| **MEDIUM_RISK** | AKIEC, BKL, VASC | 2× | 33% of Weighted F1 |
| **BENIGN** | BEN_OTH, DF, INF, NV | 1× | 17% of Weighted F1 |

### Efficiency Score

| Model Size | Efficiency Score |
|------------|-----------------|
| ≤ 50 MB | 1.0 (full points) |
| 50-150 MB | Linear decay |
| > 150 MB | 0.0 |

### Training Optimizations

This pipeline includes optimizations for high competition scores:

1. **Weighted Loss Function**: 3× weight for HIGH_RISK, 2× for MEDIUM_RISK
2. **Weighted Sampling**: 12× sampling boost for HIGH_RISK classes
3. **Data Augmentation**: Mixup, flips, rotations, color jitter
4. **Competition-Aligned Validation**: Best model saved by competition score

---

## Evaluation

### Evaluate Trained Model

```bash
python custom_model/evaluate.py \
    --model checkpoints/model.onnx \
    --data-dir training_data
```

### Compare Models

```bash
python custom_model/evaluate.py \
    --model checkpoints/model.onnx \
    --compare available_models/skin.onnx \
    --data-dir training_data
```

---

## Submission

After training, your ONNX model is ready for submission:

```bash
# Upload to HuggingFace
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='checkpoints/model.onnx',
    path_in_repo='model.onnx',
    repo_id='YOUR_USERNAME/tricorder3-model',
    repo_type='model',
)
"

# Submit to competition
PYTHONPATH="." python neurons/miner.py \
    --action submit \
    --netuid 76 \
    --hf_repo_id YOUR_USERNAME/tricorder3-model \
    --hf_model_filename model.onnx \
    --competition_id tricorder-3
```

---

## File Structure

```
custom_model/
├── train_config.yaml    # Configuration file (edit this!)
├── train.py             # Main training script
├── trainer.py           # Training logic & loss functions
├── model.py             # Model architecture
├── evaluate.py          # Evaluation script
├── download_datasets.py # Dataset download utility
└── README.md            # This file

training_data/           # Downloaded datasets
├── images/
├── labels.csv
└── labels_final.csv

checkpoints/             # Training outputs
├── best.pt              # Best model checkpoint
├── latest.pt            # Latest checkpoint (for resume)
├── checkpoint_epoch*.pt # Periodic checkpoints
└── model.onnx           # Exported ONNX model

runs/                    # TensorBoard logs
└── YYYYMMDD-HHMMSS/
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python custom_model/train.py --batch-size 8

# Or use lighter model
python custom_model/train.py --model-type lightweight
```

### CUDA Not Available

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Module Not Found

```bash
# Run from project root with PYTHONPATH
PYTHONPATH="." python custom_model/train.py

# Or on Windows
set PYTHONPATH=.
python custom_model/train.py
```

### Checkpoint Loading Error (PyTorch 2.6+)

This is automatically handled. If you still get errors:

```bash
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0
python custom_model/train.py --resume checkpoints/latest.pt
```

### TensorBoard Not Working

```bash
pip install tensorboard
tensorboard --logdir runs
```

### Windows: num_workers Error

Edit `train_config.yaml`:

```yaml
training:
  num_workers: 0  # Set to 0 on Windows
```

---

## Tips for High Scores

1. **More Data**: Larger, diverse datasets improve generalization
2. **Focus on HIGH_RISK**: These contribute 50% to weighted F1
3. **Use Balanced Model**: Best size/accuracy tradeoff (<50MB for full efficiency)
4. **Train Longer**: 50-100 epochs with patience
5. **Enable All Augmentations**: Mixup, weighted sampling
6. **Monitor TensorBoard**: Watch for overfitting

---

## License

This training pipeline is part of the Cancer AI (SafeScan) Bittensor subnet.
