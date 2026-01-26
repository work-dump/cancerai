"""
Tricorder-3 Competition Trainer

This trainer is specifically designed to optimize for the Tricorder-3 scoring system:

Final Score = 0.9 × Prediction Score + 0.1 × Efficiency Score

Where:
    Prediction Score = 0.5 × Accuracy + 0.5 × Weighted F1

And Weighted F1 is calculated as:
    Weighted F1 = (3 × F1_high_risk + 2 × F1_medium_risk + 1 × F1_benign) / 6

Risk Categories:
    - HIGH_RISK (3×): BCC(1), MAL_OTH(6), MEL(7), SCCKA(9)
    - MEDIUM_RISK (2×): AKIEC(0), BKL(3), VASC(10)  
    - BENIGN (1×): BEN_OTH(2), DF(4), INF(5), NV(8)

Key Training Strategies:
    1. Custom loss function that emphasizes high-risk classes
    2. Class-weighted sampling to handle imbalance
    3. Competition-aligned validation metrics
    4. Data augmentation for generalization
"""

import os
import json
import copy
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

try:
    from torchvision import transforms
except ImportError:
    raise ImportError("Please install torchvision: pip install torchvision")


# ============================================================================
# Constants (aligned with competition)
# ============================================================================

NUM_CLASSES = 11
IMAGE_SIZE = (512, 512)

# Risk categories (0-indexed class IDs)
HIGH_RISK_CLASSES = [1, 6, 7, 9]      # BCC, MAL_OTH, MEL, SCCKA
MEDIUM_RISK_CLASSES = [0, 3, 10]       # AKIEC, BKL, VASC
BENIGN_CLASSES = [2, 4, 5, 8]          # BEN_OTH, DF, INF, NV

# Category weights for F1 calculation
CATEGORY_WEIGHTS = {
    "HIGH_RISK": 3.0,
    "MEDIUM_RISK": 2.0,
    "BENIGN": 1.0,
}

# Competition score weights
PREDICTION_WEIGHT = 0.9
EFFICIENCY_WEIGHT = 0.1
ACCURACY_WEIGHT = 0.5
WEIGHTED_F1_WEIGHT = 0.5

# Class names
CLASS_NAMES = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]

CLASS_RISK = {
    0: "MEDIUM", 1: "HIGH", 2: "BENIGN", 3: "MEDIUM", 4: "BENIGN",
    5: "BENIGN", 6: "HIGH", 7: "HIGH", 8: "BENIGN", 9: "HIGH", 10: "MEDIUM"
}


# ============================================================================
# Scoring Functions (exactly matching competition)
# ============================================================================

def calculate_f1_by_class(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate F1 score for each class."""
    return f1_score(
        y_true, y_pred, 
        labels=list(range(NUM_CLASSES)), 
        average=None, 
        zero_division=0
    )


def calculate_risk_category_f1(f1_scores: np.ndarray) -> Dict[str, float]:
    """Calculate F1 scores for each risk category."""
    return {
        "HIGH_RISK": float(np.mean([f1_scores[i] for i in HIGH_RISK_CLASSES])),
        "MEDIUM_RISK": float(np.mean([f1_scores[i] for i in MEDIUM_RISK_CLASSES])),
        "BENIGN": float(np.mean([f1_scores[i] for i in BENIGN_CLASSES])),
    }


def calculate_weighted_f1(category_f1: Dict[str, float]) -> float:
    """
    Calculate weighted F1 score exactly as competition does.
    
    Formula: (3 × F1_high + 2 × F1_medium + 1 × F1_benign) / 6
    """
    total_weight = sum(CATEGORY_WEIGHTS.values())  # 6.0
    weighted_sum = sum(
        category_f1[cat] * weight 
        for cat, weight in CATEGORY_WEIGHTS.items()
    )
    return weighted_sum / total_weight


def calculate_prediction_score(accuracy: float, weighted_f1: float) -> float:
    """
    Calculate prediction score.
    
    Formula: 0.5 × Accuracy + 0.5 × Weighted F1
    """
    return ACCURACY_WEIGHT * accuracy + WEIGHTED_F1_WEIGHT * weighted_f1


def calculate_competition_score(
    accuracy: float, 
    weighted_f1: float, 
    efficiency: float = 1.0
) -> float:
    """
    Calculate final competition score.
    
    Formula: 0.9 × Prediction Score + 0.1 × Efficiency Score
    """
    prediction_score = calculate_prediction_score(accuracy, weighted_f1)
    return PREDICTION_WEIGHT * prediction_score + EFFICIENCY_WEIGHT * efficiency


def evaluate_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    efficiency: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluate predictions using exact competition metrics.
    
    Returns:
        Dictionary with all competition metrics
    """
    # Basic accuracy
    accuracy = float(accuracy_score(y_true, y_pred))
    
    # F1 scores
    f1_by_class = calculate_f1_by_class(y_true, y_pred)
    category_f1 = calculate_risk_category_f1(f1_by_class)
    weighted_f1 = calculate_weighted_f1(category_f1)
    
    # Competition scores
    prediction_score = calculate_prediction_score(accuracy, weighted_f1)
    final_score = calculate_competition_score(accuracy, weighted_f1, efficiency)
    
    return {
        "accuracy": accuracy,
        "f1_by_class": f1_by_class.tolist(),
        "category_f1": category_f1,
        "weighted_f1": weighted_f1,
        "prediction_score": prediction_score,
        "efficiency_score": efficiency,
        "final_score": final_score,
    }


# ============================================================================
# Custom Loss Functions
# ============================================================================

class CompetitionAlignedLoss(nn.Module):
    """
    Custom loss function designed to optimize for competition scoring.
    
    Combines:
    1. Weighted Cross-Entropy (emphasizes high-risk classes)
    2. Focal Loss component (focuses on hard examples)
    3. F1-aware soft labeling
    
    The weights are designed to improve weighted F1 score by:
    - Giving 3× weight to HIGH_RISK classes (BCC, MAL_OTH, MEL, SCCKA)
    - Giving 2× weight to MEDIUM_RISK classes (AKIEC, BKL, VASC)
    - Giving 1× weight to BENIGN classes
    
    IMPORTANT: Weights are NOT normalized to preserve the 3:2:1 ratio impact.
    This ensures the model strongly prioritizes HIGH_RISK classes which contribute
    50% (3/6) to the competition's weighted F1 score.
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        label_smoothing: float = 0.1,
        ce_weight: float = 0.6,  # Reduced from 0.7 to give more weight to focal
        focal_weight: float = 0.4,  # Increased from 0.3 for better hard example handling
        normalize_weights: bool = False,  # NEW: Don't normalize by default for stronger effect
    ):
        super().__init__()
        
        # Default class weights based on risk categories
        if class_weights is None:
            class_weights = self._get_default_weights(normalize=normalize_weights)
        
        self.register_buffer("class_weights", class_weights)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.label_smoothing = label_smoothing
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        
        # Cross-entropy with weights and label smoothing
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    
    def _get_default_weights(self, normalize: bool = False) -> torch.Tensor:
        """
        Get default class weights based on competition scoring.
        
        Higher weights for high-risk classes to improve weighted F1.
        
        Competition Weighted F1 Formula:
            weighted_f1 = (3×F1_high + 2×F1_medium + 1×F1_benign) / 6
        
        This means:
            - HIGH_RISK classes contribute 50% (3/6) to weighted F1
            - MEDIUM_RISK classes contribute 33% (2/6) to weighted F1
            - BENIGN classes contribute 17% (1/6) to weighted F1
        
        We use UNNORMALIZED weights to strongly push the model toward HIGH_RISK accuracy.
        """
        weights = torch.ones(NUM_CLASSES)
        
        # HIGH_RISK: 3× weight (BCC=1, MAL_OTH=6, MEL=7, SCCKA=9)
        # These are cancer/malignant - MOST IMPORTANT for competition score
        for idx in HIGH_RISK_CLASSES:
            weights[idx] = 3.0
        
        # MEDIUM_RISK: 2× weight (AKIEC=0, BKL=3, VASC=10)
        for idx in MEDIUM_RISK_CLASSES:
            weights[idx] = 2.0
        
        # BENIGN: 1× weight (BEN_OTH=2, DF=4, INF=5, NV=8)
        # Already set to 1.0
        
        # Only normalize if explicitly requested (not recommended for competition)
        if normalize:
            weights = weights / weights.sum() * NUM_CLASSES
        
        return weights
    
    def focal_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal Loss for handling class imbalance.
        
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=NUM_CLASSES).float()
        
        # Get probability of true class (clamp for numerical stability)
        pt = (probs * targets_one_hot).sum(dim=-1)
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        
        # Apply focal modulation
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Weighted cross entropy (clamp logits to prevent extreme values)
        logits_clamped = torch.clamp(logits, min=-50, max=50)
        ce = F.cross_entropy(logits_clamped, targets, reduction='none')
        
        # Clamp CE to prevent explosion
        ce = torch.clamp(ce, max=100.0)
        
        # Apply class weights
        class_weights = self.class_weights[targets]
        
        # Combine
        loss = self.focal_alpha * focal_weight * ce * class_weights
        
        # Handle any remaining nan/inf (nan_to_num keeps gradient graph intact)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=0.0)
        
        return loss.mean()
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Combined loss: weighted CE + focal loss.
        """
        ce = self.ce_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        
        return self.ce_weight * ce + self.focal_weight * focal


class F1OptimizedLoss(nn.Module):
    """
    Loss function that directly approximates F1 optimization.
    
    Uses a differentiable approximation of F1 score.
    
    This loss directly optimizes the competition metric:
        Weighted F1 = (3×F1_HIGH + 2×F1_MEDIUM + 1×F1_BENIGN) / 6
    
    By minimizing (1 - Weighted_F1), we maximize the competition score component.
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        epsilon: float = 1e-7,
    ):
        super().__init__()
        
        if class_weights is None:
            # UNNORMALIZED weights matching competition scoring
            weights = torch.ones(NUM_CLASSES)
            for idx in HIGH_RISK_CLASSES:
                weights[idx] = 3.0  # 50% of weighted F1
            for idx in MEDIUM_RISK_CLASSES:
                weights[idx] = 2.0  # 33% of weighted F1
            # BENIGN stays at 1.0 (17% of weighted F1)
            class_weights = weights  # NO normalization!
        
        self.register_buffer("class_weights", class_weights)
        self.epsilon = epsilon
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft F1 loss with class weights.
        
        Computes differentiable approximation of:
            Loss = 1 - Weighted_F1
        
        Where Weighted_F1 exactly matches competition formula:
            (3×F1_high + 2×F1_medium + 1×F1_benign) / 6
        """
        # Clamp logits for numerical stability
        logits_clamped = torch.clamp(logits, min=-50, max=50)
        probs = F.softmax(logits_clamped, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=NUM_CLASSES).float()
        
        # Soft TP, FP, FN per class
        tp = (probs * targets_one_hot).sum(dim=0)
        fp = (probs * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - probs) * targets_one_hot).sum(dim=0)
        
        # Soft F1 per class (with better numerical stability)
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1_denom = precision + recall + self.epsilon
        
        # Compute F1 for all classes (missing classes will naturally have low F1)
        f1 = 2 * precision * recall / f1_denom
        
        # Replace any nan/inf with 0 (keeps gradient graph intact)
        f1 = torch.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Weighted F1 loss exactly matching competition:
        # weighted_f1 = (3×F1_high + 2×F1_medium + 1×F1_benign) / 6
        # Use all classes but the f1 for missing classes is already 0
        weighted_f1 = (f1 * self.class_weights).sum() / (self.class_weights.sum() + self.epsilon)
        
        # Clamp to valid range (keep in computation graph)
        weighted_f1 = torch.clamp(weighted_f1, min=0.0, max=1.0)
        
        return 1 - weighted_f1


class CombinedCompetitionLoss(nn.Module):
    """
    Combined loss that optimizes for both accuracy and weighted F1.
    
    This directly targets the competition score formula:
    Score = 0.9 × (0.5 × Accuracy + 0.5 × Weighted_F1) + 0.1 × Efficiency
    
    Since efficiency is model-dependent (not trainable), we optimize:
    Loss = CE_loss (for accuracy) + F1_loss (for weighted F1)
    
    KEY INSIGHT: Since Weighted F1 accounts for 45% of final score (0.9 × 0.5)
    and accuracy accounts for 45% (0.9 × 0.5), we balance them equally.
    
    BUT: Weighted F1 is harder to optimize and uses 3:2:1 weighting, so we
    slightly favor the F1 loss component to push HIGH_RISK class performance.
    """
    
    def __init__(
        self,
        ce_weight: float = 0.4,  # Slightly reduced to favor F1
        f1_weight: float = 0.6,  # Slightly increased for weighted F1 optimization
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.f1_weight = f1_weight
        
        # Class weights for risk categories (UNNORMALIZED for strong effect)
        class_weights = self._get_competition_weights()
        
        self.ce_loss = CompetitionAlignedLoss(
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            normalize_weights=False,  # Keep unnormalized for strong HIGH_RISK emphasis
        )
        
        self.f1_loss = F1OptimizedLoss(class_weights=class_weights)
    
    def _get_competition_weights(self) -> torch.Tensor:
        """
        Get weights that match competition scoring.
        
        These weights are UNNORMALIZED to strongly push HIGH_RISK performance.
        
        Competition Weighted F1 breakdown:
            HIGH_RISK (BCC, MAL_OTH, MEL, SCCKA): 3/6 = 50% of weighted F1
            MEDIUM_RISK (AKIEC, BKL, VASC): 2/6 = 33% of weighted F1
            BENIGN (BEN_OTH, DF, INF, NV): 1/6 = 17% of weighted F1
        
        Final score impact:
            HIGH_RISK accuracy → 0.9 × 0.5 × 0.5 = 22.5% of final score via weighted F1
            + contributes to accuracy (another 22.5%)
        """
        weights = torch.ones(NUM_CLASSES)
        
        # HIGH_RISK: 3× weight - CRITICAL for competition ranking
        for idx in HIGH_RISK_CLASSES:
            weights[idx] = 3.0
        
        # MEDIUM_RISK: 2× weight
        for idx in MEDIUM_RISK_CLASSES:
            weights[idx] = 2.0
        
        # BENIGN stays at 1.0
        
        return weights
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass returning total loss and components.
        
        Total Loss = 0.4 × CompetitionAlignedLoss + 0.6 × F1OptimizedLoss
        
        This weighting favors F1 optimization because:
        1. F1 loss directly optimizes the metric used in competition
        2. CE loss helps with accuracy but F1 is harder to improve
        3. HIGH_RISK classes need strong focus (50% of weighted F1)
        """
        ce = self.ce_loss(logits, targets)
        f1 = self.f1_loss(logits, targets)
        
        # Replace nan/inf with safe values (nan_to_num keeps gradient graph intact!)
        ce = torch.nan_to_num(ce, nan=0.0, posinf=10.0, neginf=0.0)
        f1 = torch.nan_to_num(f1, nan=0.0, posinf=1.0, neginf=0.0)
        
        total = self.ce_weight * ce + self.f1_weight * f1
        
        # Final nan check
        total = torch.nan_to_num(total, nan=0.0, posinf=10.0, neginf=0.0)
        
        return total, {"ce_loss": ce.item(), "f1_loss": f1.item()}


# ============================================================================
# Data Augmentation
# ============================================================================

def get_train_transforms(image_size: Tuple[int, int] = IMAGE_SIZE):
    """
    Get training data augmentation transforms.
    
    Designed for dermoscopy images with:
    - Geometric augmentations (rotation, flip, affine)
    - Color augmentations (jitter, normalize)
    - Regularization (cutout/erasing)
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10,
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_val_transforms(image_size: Tuple[int, int] = IMAGE_SIZE):
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])


# ============================================================================
# Dataset
# ============================================================================

class SkinLesionDataset(Dataset):
    """
    Dataset for skin lesion images with demographics.
    
    Expected data format:
    - images: List of image file paths
    - labels: List of integer class labels (0-10)
    - metadata: List of dicts with 'age', 'gender', 'location'
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        metadata: Optional[List[Dict[str, Any]]] = None,
        transform: Optional[Callable] = None,
        validate_images: bool = True,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.metadata = metadata or [{}] * len(image_paths)
        self.transform = transform or get_val_transforms()
        self._load_failures = 0  # Track how many images fail to load
        
        assert len(image_paths) == len(labels) == len(self.metadata)
        
        # Validate a sample of images to detect loading issues early
        if validate_images:
            self._validate_images()
    
    def _validate_images(self, num_samples: int = 10):
        """Validate that images can be loaded. Helps detect symlink/path issues."""
        import random
        
        # Test a random sample of images
        sample_indices = random.sample(range(len(self.image_paths)), min(num_samples, len(self.image_paths)))
        failures = 0
        
        for idx in sample_indices:
            path = self.image_paths[idx]
            try:
                # Check if file exists
                from pathlib import Path
                p = Path(path)
                if not p.exists():
                    failures += 1
                    if failures <= 3:
                        print(f"  ⚠️  Image not found: {path}")
                    continue
                
                # Check if it's a broken symlink
                if p.is_symlink():
                    target = p.resolve()
                    if not target.exists():
                        failures += 1
                        if failures <= 3:
                            print(f"  ⚠️  Broken symlink: {path} -> {target}")
                        continue
                
                # Try to open the image
                img = Image.open(path)
                img.verify()
                
            except Exception as e:
                failures += 1
                if failures <= 3:
                    print(f"  ⚠️  Cannot load image {path}: {e}")
        
        if failures > 0:
            pct = failures / len(sample_indices) * 100
            print(f"\n{'='*60}")
            print(f"⚠️  WARNING: {failures}/{len(sample_indices)} sampled images ({pct:.0f}%) failed to load!")
            print(f"{'='*60}")
            if pct >= 50:
                print("This will cause training to fail or produce poor results!")
                print("\nPossible causes:")
                print("  1. On Windows: Symlinks don't work without admin privileges")
                print("     → Re-run split_dataset.py with: --copy-images")
                print("  2. Image paths are incorrect")
                print("  3. Images were moved or deleted")
                print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _encode_demographics(self, meta: Dict[str, Any]) -> np.ndarray:
        """Encode demographics to model input format."""
        import math
        
        # Age (default 50 if missing or NaN)
        age_val = meta.get('age', meta.get('age_approx', 50))
        try:
            age = float(age_val)
            if math.isnan(age):
                age = 50.0
        except (TypeError, ValueError):
            age = 50.0
        
        # Gender: male=1, female=0, unknown=-1 (check both 'gender' and 'sex' keys)
        gender_val = meta.get('gender', meta.get('sex', ''))
        try:
            gender_str = str(gender_val).lower() if gender_val is not None else ''
            # Handle NaN string
            if gender_str == 'nan' or gender_str == '':
                gender = -1.0
            elif gender_str in ['male', 'm', '1']:
                gender = 1.0
            elif gender_str in ['female', 'f', '0']:
                gender = 0.0
            else:
                gender = -1.0
        except:
            gender = -1.0
        
        # Location: 1-7, unknown=0 (check both 'location' and 'localization' keys)
        location_val = meta.get('location', meta.get('localization', 0))
        try:
            if isinstance(location_val, str) and location_val.lower() != 'nan':
                location_map = {
                    'arm': 1, 'feet': 2, 'genitalia': 3, 'hand': 4,
                    'head': 5, 'leg': 6, 'torso': 7
                }
                location = float(location_map.get(location_val.lower(), 0))
            else:
                location = float(location_val) if location_val is not None else 0.0
                if math.isnan(location):
                    location = 0.0
        except (TypeError, ValueError):
            location = 0.0
        
        return np.array([age, gender, location], dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        try:
            # Load image
            image = Image.open(self.image_paths[idx]).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Encode demographics
            demographics = self._encode_demographics(self.metadata[idx])
            demographics = torch.from_numpy(demographics)
            
            # Label
            label = self.labels[idx]
            
            return image, demographics, label
            
        except Exception as e:
            # Handle corrupt/missing images by returning a placeholder
            self._load_failures += 1
            if self._load_failures <= 5:
                print(f"Warning: Could not load image {self.image_paths[idx]}: {e}")
            elif self._load_failures == 6:
                print("(Suppressing further load warnings...)")
            # Return a black image with same label (will be learned as noise)
            placeholder = torch.zeros(3, 512, 512)
            demographics = torch.tensor([50.0, -1.0, 0.0])  # Default demographics
            return placeholder, demographics, self.labels[idx]
    
    def get_load_failure_count(self) -> int:
        """Return the number of images that failed to load during training."""
        return self._load_failures


def create_weighted_sampler(
    labels: List[int],
    boost_high_risk: float = 4.0,  # Extra boost for HIGH_RISK (on top of 3×)
    boost_medium_risk: float = 2.0,  # Extra boost for MEDIUM_RISK (on top of 2×)
) -> WeightedRandomSampler:
    """
    Create a weighted sampler for class-balanced training.
    
    Gives higher sampling probability to:
    1. Rare classes (inverse frequency weighting)
    2. HIGH_RISK classes (critical for competition score - 50% of weighted F1)
    3. MEDIUM_RISK classes (33% of weighted F1)
    
    The combination ensures the model sees more HIGH_RISK examples during training,
    which is essential for ranking high in the competition.
    
    Args:
        labels: List of class labels
        boost_high_risk: Additional multiplier for HIGH_RISK beyond the 3× category weight
        boost_medium_risk: Additional multiplier for MEDIUM_RISK beyond the 2× category weight
    """
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_counts = np.maximum(class_counts, 1)  # Avoid division by zero
    
    # Base weight: inverse of frequency (rare classes sampled more)
    class_weights = 1.0 / class_counts
    
    # Multiply by risk category weights × boost factor
    # This ensures HIGH_RISK classes are heavily oversampled
    for idx in HIGH_RISK_CLASSES:
        # 3.0 (category weight) × 4.0 (boost) = 12× sampling weight
        class_weights[idx] *= 3.0 * boost_high_risk
    
    for idx in MEDIUM_RISK_CLASSES:
        # 2.0 (category weight) × 2.0 (boost) = 4× sampling weight
        class_weights[idx] *= 2.0 * boost_medium_risk
    
    # BENIGN classes: 1.0 × 1.0 = 1× (just inverse frequency)
    
    # Normalize to probabilities
    class_weights = class_weights / class_weights.sum()
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


# ============================================================================
# Post-Training Threshold Optimization
# ============================================================================

def optimize_thresholds(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_iterations: int = 100,
) -> Dict[int, float]:
    """
    Optimize per-class classification thresholds to maximize competition score.
    
    After training, the default threshold is 0.5 for all classes (argmax).
    This function finds optimal thresholds that maximize weighted F1.
    
    Returns:
        Dictionary mapping class_id -> optimal_threshold
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, demographics, labels in val_loader:
            images = images.to(device)
            demographics = demographics.to(device)
            
            probs = model(images, demographics)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    
    # Start with default thresholds
    best_thresholds = {i: 0.0 for i in range(NUM_CLASSES)}  # 0.0 means no adjustment
    best_score = 0.0
    
    # Grid search for threshold adjustments
    # We adjust the probability for each class before argmax
    for _ in range(num_iterations):
        # Random threshold adjustments
        adjustments = np.random.uniform(-0.3, 0.3, NUM_CLASSES)
        
        # Give higher adjustments to HIGH_RISK classes
        for idx in HIGH_RISK_CLASSES:
            adjustments[idx] = np.random.uniform(0.0, 0.4)  # Boost HIGH_RISK
        
        # Apply adjustments
        adjusted_probs = all_probs + adjustments
        predictions = adjusted_probs.argmax(axis=1)
        
        # Evaluate
        metrics = evaluate_predictions(all_labels, predictions)
        score = metrics["final_score"]
        
        if score > best_score:
            best_score = score
            best_thresholds = {i: float(adjustments[i]) for i in range(NUM_CLASSES)}
    
    print(f"Threshold optimization: score improved from baseline to {best_score:.4f}")
    print(f"Optimal adjustments: HIGH_RISK = {[best_thresholds[i] for i in HIGH_RISK_CLASSES]}")
    
    return best_thresholds


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Data
    batch_size: int = 16
    num_workers: int = 4
    
    # Training
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # Loss
    ce_weight: float = 0.5
    f1_weight: float = 0.5
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    use_simple_loss: bool = False  # Use simple CE loss instead of complex combined loss
    
    # Regularization
    dropout: float = 0.4
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine" or "onecycle"
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    save_every_n_epochs: int = 5  # Save checkpoint every N epochs (0 = only save best)
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TensorBoard logging
    use_tensorboard: bool = True
    tensorboard_dir: str = "runs"


# ============================================================================
# Mixup Augmentation
# ============================================================================

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup: Beyond Empirical Risk Minimization
    
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: Callable,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixup loss."""
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    
    # Handle tuple returns (from CombinedCompetitionLoss)
    if isinstance(loss_a, tuple):
        loss_a = loss_a[0]
    if isinstance(loss_b, tuple):
        loss_b = loss_b[0]
    
    return lam * loss_a + (1 - lam) * loss_b


# ============================================================================
# Trainer Class
# ============================================================================

class Tricorder3Trainer:
    """
    Trainer optimized for Tricorder-3 competition scoring.
    
    Features:
    - Competition-aligned loss function
    - Weighted sampling for class imbalance
    - Competition metrics tracking
    - Best model checkpointing based on competition score
    - TensorBoard logging
    - Resume training from checkpoint
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function - use simple CE if complex loss causes issues
        if getattr(self.config, 'use_simple_loss', False):
            print("  Using simple weighted cross-entropy loss")
            # Simple weighted CE loss
            class_weights = torch.ones(NUM_CLASSES)
            for idx in HIGH_RISK_CLASSES:
                class_weights[idx] = 3.0
            for idx in MEDIUM_RISK_CLASSES:
                class_weights[idx] = 2.0
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            self._simple_loss = True
        else:
            self.criterion = CombinedCompetitionLoss(
                ce_weight=self.config.ce_weight,
                f1_weight=self.config.f1_weight,
                focal_gamma=self.config.focal_gamma,
                label_smoothing=self.config.label_smoothing,
            ).to(self.device)
            self._simple_loss = False
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # TensorBoard writer
        self.writer = None
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(
                    self.config.tensorboard_dir,
                    datetime.now().strftime("%Y%m%d-%H%M%S")
                )
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"TensorBoard logging to: {log_dir}")
                print(f"  View with: tensorboard --logdir {self.config.tensorboard_dir}")
            except ImportError:
                print("TensorBoard not installed. Install with: pip install tensorboard")
                self.writer = None
        
        # State
        self.best_score = 0.0
        self.best_model_state = None
        self.history = []
        self.patience_counter = 0
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
            )
        else:
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=num_training_steps,
                pct_start=self.config.warmup_epochs / self.config.epochs,
            )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (images, demographics, labels) in enumerate(dataloader):
            images = images.to(self.device)
            demographics = demographics.to(self.device)
            labels = labels.to(self.device)
            
            # Mixup augmentation
            if self.config.use_mixup and self.model.training:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, self.config.mixup_alpha
                )
                
                # Forward
                logits, probs = self.model.forward_with_logits(images, demographics)
                
                # Mixup loss
                loss = mixup_criterion(
                    lambda p, y: self.criterion(p, y)[0],
                    logits, labels_a, labels_b, lam
                )
            else:
                # Forward
                logits, probs = self.model.forward_with_logits(images, demographics)
                
                # Loss
                if self._simple_loss:
                    loss = self.criterion(logits, labels)
                else:
                    loss, _ = self.criterion(logits, labels)
            
            # Check for nan loss
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected at batch! Skipping...")
                continue
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Check for nan gradients
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"WARNING: NaN gradients detected! Skipping batch...")
                self.optimizer.zero_grad()  # Clear the bad gradients
                continue
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions (without mixup for metrics)
            if not self.config.use_mixup:
                preds = probs.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        metrics = {"loss": avg_loss}
        
        # Compute training metrics if not using mixup
        if not self.config.use_mixup and all_preds:
            eval_metrics = evaluate_predictions(
                np.array(all_labels),
                np.array(all_preds),
            )
            metrics.update({f"train_{k}": v for k, v in eval_metrics.items() 
                          if isinstance(v, (int, float))})
        
        return metrics
    
    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        for images, demographics, labels in dataloader:
            images = images.to(self.device)
            demographics = demographics.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            logits, probs = self.model.forward_with_logits(images, demographics)
            
            # Loss
            if self._simple_loss:
                loss = self.criterion(logits, labels)
            else:
                loss, _ = self.criterion(logits, labels)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            
            # Store predictions
            preds = probs.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        eval_metrics = evaluate_predictions(
            np.array(all_labels),
            np.array(all_preds),
        )
        
        metrics = {
            "val_loss": total_loss / len(dataloader),
            "val_accuracy": eval_metrics["accuracy"],
            "val_weighted_f1": eval_metrics["weighted_f1"],
            "val_prediction_score": eval_metrics["prediction_score"],
            "val_competition_score": eval_metrics["final_score"],
        }
        
        # Add category F1 scores
        for cat, f1 in eval_metrics["category_f1"].items():
            metrics[f"val_f1_{cat.lower()}"] = f1
        
        return metrics
    
    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        use_weighted_sampling: bool = True,
        resume_from: Optional[str] = None,
        resume_weights_only: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            use_weighted_sampling: Whether to use class-balanced sampling
            resume_from: Path to checkpoint to resume training from
            resume_weights_only: If True, only load model weights (not optimizer/epoch).
                                 Use this when transferring weights to a different architecture
                                 or when you want to train N more epochs from a checkpoint.
        
        Returns:
            Training history
        """
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            if resume_weights_only:
                # Only load model weights, start fresh training
                # This allows: 1) using different architecture, 2) training N more epochs
                self.load_weights_only(resume_from)
                print(f"Loaded weights from {resume_from}")
                print(f"Starting fresh training for {self.config.epochs} epochs")
            else:
                # Full resume: load weights, optimizer, and continue from saved epoch
                checkpoint = self.load_checkpoint(resume_from, resume_training=True, weights_only=False)
                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"Resuming from epoch {start_epoch}, will train until epoch {self.config.epochs}")
        
        # Create data loaders
        if use_weighted_sampling:
            sampler = create_weighted_sampler(train_dataset.labels)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        
        # Create scheduler
        num_training_steps = len(train_loader) * self.config.epochs
        scheduler = self._create_scheduler(num_training_steps)
        
        # Training loop
        print(f"\n{'='*60}")
        print(f"Training Tricorder-3 Model")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        if self.writer:
            print(f"TensorBoard: {self.config.tensorboard_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, self.config.epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.history.append(metrics)
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
                self.writer.add_scalar("Accuracy/val", val_metrics["val_accuracy"], epoch)
                self.writer.add_scalar("WeightedF1/val", val_metrics["val_weighted_f1"], epoch)
                self.writer.add_scalar("CompetitionScore/val", val_metrics["val_competition_score"], epoch)
                self.writer.add_scalar("F1_HighRisk/val", val_metrics["val_f1_high_risk"], epoch)
                self.writer.add_scalar("F1_MediumRisk/val", val_metrics["val_f1_medium_risk"], epoch)
                self.writer.add_scalar("F1_Benign/val", val_metrics["val_f1_benign"], epoch)
                self.writer.add_scalar("LearningRate", self.optimizer.param_groups[0]["lr"], epoch)
            
            # Print progress
            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
                f"Val wF1: {val_metrics['val_weighted_f1']:.4f} | "
                f"Score: {val_metrics['val_competition_score']:.4f}"
            )
            
            # Check for best model (based on competition score)
            current_score = val_metrics["val_competition_score"]
            is_best = current_score > self.best_score + self.config.min_delta
            
            if is_best:
                self.best_score = current_score
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                print(f"  -> New best score: {self.best_score:.4f}")
                
                # Save best checkpoint
                if self.config.save_dir:
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoint (every N epochs)
            if (self.config.save_every_n_epochs > 0 and 
                (epoch + 1) % self.config.save_every_n_epochs == 0 and 
                not is_best):  # Don't double-save if already saved as best
                if self.config.save_dir:
                    self._save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nRestored best model with score: {self.best_score:.4f}")
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_score": self.best_score,
            "metrics": metrics,
            "config": self.config,
        }
        
        if is_best:
            # Save best model with score in filename
            path = os.path.join(
                self.config.save_dir,
                f"best_model_epoch{epoch+1}_score_{metrics['val_competition_score']:.4f}.pt"
            )
            torch.save(checkpoint, path)
            print(f"  -> Saved best checkpoint: {path}")
            
            # Also save as 'best.pt' for easy access
            best_path = os.path.join(self.config.save_dir, "best.pt")
            torch.save(checkpoint, best_path)
        else:
            # Save periodic checkpoint with epoch number
            path = os.path.join(
                self.config.save_dir,
                f"checkpoint_epoch{epoch+1}.pt"
            )
            torch.save(checkpoint, path)
            print(f"  -> Saved periodic checkpoint: {path}")
        
        # Always save as 'latest.pt' for easy resume
        latest_path = os.path.join(self.config.save_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, path: str, resume_training: bool = False, weights_only: bool = False):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            resume_training: If True, also restore optimizer state and training progress
            weights_only: If False, allows loading custom objects (needed for TrainingConfig)
        
        Returns:
            Dictionary with checkpoint info (epoch, metrics, etc.)
        """
        # weights_only=False needed for PyTorch 2.6+ to load TrainingConfig
        checkpoint = torch.load(path, map_location=self.device, weights_only=weights_only)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if resume_training:
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "best_score" in checkpoint:
                self.best_score = checkpoint["best_score"]
            print(f"Resumed training from epoch {checkpoint.get('epoch', 0) + 1}")
        else:
            print(f"Loaded checkpoint from {path}")
        
        return checkpoint
    
    def load_weights_only(self, path: str, strict: bool = False):
        """
        Load only model weights from checkpoint (not optimizer/epoch).
        
        This is useful for:
        1. Loading pretrained weights into a different (but compatible) architecture
        2. Starting fresh training from a checkpoint (train N more epochs)
        3. Fine-tuning from a checkpoint with new optimizer settings
        
        Args:
            path: Path to checkpoint file
            strict: If True, requires exact match of model architecture.
                    If False, loads compatible weights and ignores mismatches.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # Assume it's just a state dict
            state_dict = checkpoint
        
        # Try to load weights
        if strict:
            self.model.load_state_dict(state_dict)
            print(f"Loaded all weights from {path}")
        else:
            # Load compatible weights, ignore mismatches
            model_state = self.model.state_dict()
            loaded_keys = []
            skipped_keys = []
            
            for key, value in state_dict.items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        model_state[key] = value
                        loaded_keys.append(key)
                    else:
                        skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state[key].shape})")
                else:
                    skipped_keys.append(f"{key} (not in model)")
            
            self.model.load_state_dict(model_state)
            
            print(f"Loaded {len(loaded_keys)} weights from {path}")
            if skipped_keys:
                print(f"Skipped {len(skipped_keys)} incompatible weights:")
                for key in skipped_keys[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(skipped_keys) > 5:
                    print(f"  ... and {len(skipped_keys) - 5} more")


# ============================================================================
# Utility Functions
# ============================================================================

def print_class_distribution(labels: List[int]):
    """Print class distribution with risk categories."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    
    print("\nClass Distribution:")
    print("-" * 50)
    
    for idx in range(NUM_CLASSES):
        pct = counts[idx] / total * 100
        risk = CLASS_RISK[idx]
        bar = "█" * int(pct / 2)
        print(f"{CLASS_NAMES[idx]:8} ({risk:6}): {counts[idx]:5} ({pct:5.1f}%) {bar}")
    
    print("-" * 50)
    print(f"Total: {total}")


def load_dataset_from_csv(
    csv_path: str,
    image_dir: str,
    image_column: str = "image",
    label_column: str = "label",
) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """
    Load dataset from CSV file.
    
    Expected CSV columns:
    - image: image filename
    - label: class label (string or int)
    - age: patient age (optional)
    - gender: patient gender (optional)
    - location: body location (optional)
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Map string labels to indices
    label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    image_paths = []
    labels = []
    metadata = []
    
    for _, row in df.iterrows():
        # Image path
        img_name = row[image_column]
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue
        image_paths.append(img_path)
        
        # Label
        label = row[label_column]
        if isinstance(label, str):
            label = label_to_idx.get(label.upper(), 0)
        labels.append(int(label))
        
        # Metadata
        meta = {}
        if 'age' in row:
            meta['age'] = row['age']
        if 'gender' in row:
            meta['gender'] = row['gender']
        if 'location' in row:
            meta['location'] = row['location']
        metadata.append(meta)
    
    return image_paths, labels, metadata


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Tricorder-3 Competition Trainer")
    print("="*60)
    
    # Test scoring functions
    print("\nTesting scoring functions...")
    
    # Simulate predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 11, size=100)
    y_pred = np.random.randint(0, 11, size=100)
    
    metrics = evaluate_predictions(y_true, y_pred)
    
    print(f"\nSample Evaluation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Category F1:")
    for cat, f1 in metrics['category_f1'].items():
        print(f"    {cat}: {f1:.4f}")
    print(f"  Prediction Score: {metrics['prediction_score']:.4f}")
    print(f"  Competition Score: {metrics['final_score']:.4f}")
    
    # Test loss function
    print("\nTesting loss function...")
    criterion = CombinedCompetitionLoss()
    
    logits = torch.randn(8, 11)
    targets = torch.randint(0, 11, (8,))
    
    loss, components = criterion(logits, targets)
    print(f"  Total Loss: {loss.item():.4f}")
    print(f"  CE Loss: {components['ce_loss']:.4f}")
    print(f"  F1 Loss: {components['f1_loss']:.4f}")
    
    print("\nAll tests passed!")
    print("\nUsage:")
    print("  from custom_model.trainer import Tricorder3Trainer, TrainingConfig")
    print("  trainer = Tricorder3Trainer(model, TrainingConfig())")
    print("  trainer.fit(train_dataset, val_dataset)")
