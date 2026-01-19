# Tricorder-3 Competition Training Guide

This guide will help you train a high-scoring model for the Tricorder-3 competition.

## Prerequisites

### 1. Install Required Packages

```bash
# Activate your conda environment
conda activate cancerai

# Install required packages
pip install datasets huggingface_hub tqdm scikit-learn pillow pandas torchvision timm
```

### 2. (Optional) Setup Kaggle API

For access to HAM10000 and other Kaggle datasets:

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json` to `~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

Then install:
```bash
pip install kaggle
```

## Step 1: Download Training Data

### Option A: Automatic Download (Recommended)

```bash
cd /home/ubuntu/Projects/bittensor/cancer-ai

# Download all available datasets
python custom_model/download_datasets.py --output-dir training_data
```

This will:
- Download datasets from HuggingFace
- Download from Kaggle (if configured)
- Map all labels to Tricorder-3's 11 classes
- Create synthetic data for missing classes
- Balance the dataset

### Option B: Manual Download

If automatic download fails, manually download:

1. **HAM10000** (Essential - 10,015 images):
   - https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
   - Download and extract to `training_data/ham10000/`

2. **ISIC 2019** (Additional 25K images):
   - https://challenge2019.isic-archive.com/data.html
   - Download and extract to `training_data/isic2019/`

Then run the preparation script:
```bash
python custom_model/download_datasets.py --output-dir training_data --skip-kaggle
```

## Step 2: Train the Model

### Quick Training (Test Run)
```bash
python custom_model/train.py \
    --data-dir training_data \
    --epochs 10 \
    --batch-size 16 \
    --model-type balanced
```

### Full Training (For Competition)
```bash
python custom_model/train.py \
    --data-dir training_data \
    --epochs 50 \
    --batch-size 16 \
    --model-type balanced \
    --learning-rate 1e-3 \
    --use-mixup \
    --weighted-sampling \
    --patience 15
```

### High Accuracy Training (If you have GPU and time)
```bash
python custom_model/train.py \
    --data-dir training_data \
    --epochs 100 \
    --batch-size 32 \
    --model-type balanced \
    --learning-rate 5e-4 \
    --use-mixup \
    --weighted-sampling \
    --patience 20
```

## Step 3: Evaluate Your Model

```bash
# Evaluate on validation data
python custom_model/evaluate.py \
    --model checkpoints/model.onnx \
    --data-dir training_data

# Compare with existing skin.onnx
python custom_model/evaluate.py \
    --model checkpoints/model.onnx \
    --data-dir training_data \
    --compare available_models/skin.onnx
```

## Step 4: Submit Your Model

### Upload to HuggingFace

1. Create a HuggingFace account: https://huggingface.co/join
2. Create a new model repository
3. Upload your ONNX model:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="checkpoints/model.onnx",
    path_in_repo="model.onnx",
    repo_id="YOUR_USERNAME/tricorder3-model",
    repo_type="model",
)
```

### Submit to Competition

```bash
cd /home/ubuntu/Projects/bittensor/cancer-ai

PYTHONPATH="." python neurons/miner.py \
    --action submit \
    --netuid 76 \
    --hf_repo_id YOUR_USERNAME/tricorder3-model \
    --hf_model_filename model.onnx \
    --competition_id tricorder-3
```

## Scoring System

Your model is scored as:

```
Final Score = 0.9 × Prediction Score + 0.1 × Efficiency Score

Prediction Score = 0.5 × Accuracy + 0.5 × Weighted F1

Weighted F1 = (3 × F1_HIGH_RISK + 2 × F1_MEDIUM_RISK + 1 × F1_BENIGN) / 6
```

### Risk Categories

| Category | Classes | Weight |
|----------|---------|--------|
| HIGH_RISK | BCC, MAL_OTH, MEL, SCCKA | 3× |
| MEDIUM_RISK | AKIEC, BKL, VASC | 2× |
| BENIGN | BEN_OTH, DF, INF, NV | 1× |

### Model Size

| Size | Efficiency Score |
|------|-----------------|
| ≤ 50 MB | 1.0 |
| 50-150 MB | Linear decay |
| > 150 MB | 0.0 |

## Tips for High Scores

1. **Use More Data**: More training data = better generalization
2. **Focus on HIGH_RISK Classes**: They contribute 50% to weighted F1
3. **Handle Class Imbalance**: Use weighted sampling and class-weighted loss
4. **Data Augmentation**: Helps with generalization
5. **Keep Model Small**: <50 MB for full efficiency score
6. **Ensemble**: Train multiple models and average predictions

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 8`
- Use lighter model: `--model-type lightweight`

### Training Too Slow
- Reduce epochs: `--epochs 20`
- Use fewer workers: `--num-workers 2`

### Poor Accuracy on HIGH_RISK Classes
- Increase epochs
- Ensure training data includes these classes
- Check class weights in loss function

## File Structure

```
custom_model/
├── model.py           # Model architecture
├── trainer.py         # Training logic with competition loss
├── train.py           # Main training script
├── evaluate.py        # Evaluation script
├── download_datasets.py  # Dataset download
└── TRAINING_GUIDE.md  # This guide
```
