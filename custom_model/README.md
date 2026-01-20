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
python custom_model/download_datasets.py

# 3. Train
python custom_model/train.py

# 4. Submit (model at: checkpoints/model.onnx)
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

### Step 2: Configure Datasets

Edit `train_config.yaml`:

```yaml
download:
  output_dir: "training_data"
  
  kaggle:
    ham10000: true          # 10,015 images - Essential!
    isic_2019: true         # 25,331 images - Recommended
    isic_2020: true         # 33,126 images - More melanoma data
    skin_cancer_9class: true # 9-class extended dataset
    dermnet: false          # Large dataset (optional)
  
  balance_dataset: true
  max_per_class: 3000
  min_per_class: 500
```

### Step 3: Download Datasets

```bash
python custom_model/download_datasets.py
```

Or download specific dataset:

```bash
python custom_model/download_datasets.py --dataset ham10000
```

List available datasets:

```bash
python custom_model/download_datasets.py --list
```

### Available Datasets

**Total potential: 100,000+ images from multiple sources**

#### Kaggle Datasets (requires API setup)

| Key | Dataset | Images | Description |
|-----|---------|--------|-------------|
| `ham10000` | HAM10000 | 10,015 | **Essential** - 7 classes, primary dataset |
| `isic_2019` | ISIC 2019 | 25,331 | 8 diagnostic categories |
| `isic_2020` | ISIC 2020 | 33,126 | Melanoma detection |
| `isic_2018` | ISIC 2018 | ~10,000 | Challenge dataset |
| `skin_cancer_9class` | 9-Class | ~11,000 | Extended classification |
| `dermnet` | DermNet | ~23,000 | 23 disease classes |

#### HuggingFace Datasets (no API needed)

| Key | Dataset | Description |
|-----|---------|-------------|
| `marmal88_skin` | Skin Cancer | Classification dataset |
| `isic_hf` | ISIC (HF) | ISIC on HuggingFace |

#### Direct Downloads

| Key | Dataset | Images | Description |
|-----|---------|--------|-------------|
| `ph2` | PH2 | 200 | Melanoma, atypical/common nevi |

### Download Commands

```bash
# Download enabled datasets from config
python custom_model/download_datasets.py

# Download ALL available datasets
python custom_model/download_datasets.py --all

# Download from specific source
python custom_model/download_datasets.py --source kaggle
python custom_model/download_datasets.py --source huggingface

# Download specific dataset
python custom_model/download_datasets.py --dataset ham10000

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
  skip_kaggle: false
  create_synthetic: true
  balance_dataset: true
  max_per_class: 2000
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
