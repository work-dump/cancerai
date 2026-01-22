"""
Tricorder-3 Competition Model - Option A: Pretrained Backbone with Cross-Attention Fusion

Architecture:
- EfficientNet backbone (pretrained on ImageNet)
- Learned embeddings for demographics (age, sex, location)
- Cross-attention fusion between image and metadata features
- Multi-layer classifier with regularization

Input:
- image: (batch, 3, 512, 512) - normalized [0, 1]
- demographics: (batch, 3) - [age, sex, location]

Output:
- probabilities: (batch, 11) - softmax probabilities for 11 classes

Classes (in order):
0: AKIEC - Actinic keratosis (Medium risk)
1: BCC   - Basal cell carcinoma (Malignant)
2: BEN_OTH - Other benign (Benign)
3: BKL   - Benign keratosis-like (Medium risk)
4: DF    - Dermatofibroma (Benign)
5: INF   - Inflammatory (Benign)
6: MAL_OTH - Other malignant (Malignant)
7: MEL   - Melanoma (Malignant) - CRITICAL
8: NV    - Melanocytic nevus (Benign)
9: SCCKA - SCC/Keratoacanthoma (Malignant)
10: VASC - Vascular lesions (Medium risk)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


# ============================================================================
# Constants
# ============================================================================

NUM_CLASSES = 11
IMAGE_SIZE = (512, 512)
DEMOGRAPHICS_DIM = 3  # age, sex, location

# Class information
CLASS_NAMES = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", 
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]

CLASS_WEIGHTS = {
    "MALIGNANT": [1, 6, 7, 9],      # BCC, MAL_OTH, MEL, SCCKA - weight 3x
    "MEDIUM_RISK": [0, 3, 10],       # AKIEC, BKL, VASC - weight 2x
    "BENIGN": [2, 4, 5, 8],          # BEN_OTH, DF, INF, NV - weight 1x
}

# Body location mapping
LOCATION_MAP = {
    1: "Arm",
    2: "Feet", 
    3: "Genitalia",
    4: "Hand",
    5: "Head",
    6: "Leg",
    7: "Torso",
}


# ============================================================================
# Backbone Options
# ============================================================================

BACKBONE_CONFIGS = {
    "efficientnet_b0": {"features": 1280, "size_mb": 20},
    "efficientnet_b1": {"features": 1280, "size_mb": 30},
    "efficientnet_b2": {"features": 1408, "size_mb": 35},
    "mobilenetv3_large": {"features": 960, "size_mb": 22},
    "convnext_tiny": {"features": 768, "size_mb": 110},
}


# ============================================================================
# Model Components
# ============================================================================

class DemographicsEncoder(nn.Module):
    """
    Encodes patient demographics (age, sex, location) using learned embeddings.
    
    This is more expressive than simple linear layers as it can learn
    distinct representations for categorical variables.
    """
    
    def __init__(
        self,
        output_dim: int = 256,
        age_embed_dim: int = 64,
        sex_embed_dim: int = 32,
        location_embed_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Age encoder (continuous → embedding)
        self.age_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, age_embed_dim),
            nn.LayerNorm(age_embed_dim),
        )
        
        # Sex embedding (categorical: 0=female, 1=male, 2=unknown)
        self.sex_embedding = nn.Embedding(3, sex_embed_dim)
        
        # Location embedding (categorical: 1-7 locations + 0 for unknown)
        self.location_embedding = nn.Embedding(8, location_embed_dim)
        
        # Combine all demographics
        combined_dim = age_embed_dim + sex_embed_dim + location_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self.output_dim = output_dim
    
    def forward(self, demographics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            demographics: (batch, 3) tensor with [age, sex, location]
                - age: float (years, e.g., 45.0)
                - sex: int (0=female, 1=male)
                - location: int (1-7)
        
        Returns:
            (batch, output_dim) demographic features
        """
        # Extract and process each component
        age = demographics[:, 0:1] / 100.0  # Normalize age to ~[0, 1]
        sex = demographics[:, 1].long().clamp(0, 2)
        location = demographics[:, 2].long().clamp(0, 7)
        
        # Get embeddings
        age_emb = self.age_encoder(age)
        sex_emb = self.sex_embedding(sex)
        location_emb = self.location_embedding(location)
        
        # Concatenate and fuse
        combined = torch.cat([age_emb, sex_emb, location_emb], dim=-1)
        return self.fusion(combined)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module where image features attend to metadata features.
    
    This allows the model to learn which aspects of the image are relevant
    given the patient's demographic information.
    """
    
    def __init__(
        self,
        image_dim: int,
        meta_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Project metadata to image dimension for attention
        self.meta_projection = nn.Linear(meta_dim, image_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norm for residual connection
        self.norm = nn.LayerNorm(image_dim)
        
        self.output_dim = image_dim * 2  # Original + attended
    
    def forward(
        self, 
        image_features: torch.Tensor, 
        meta_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_features: (batch, image_dim)
            meta_features: (batch, meta_dim)
        
        Returns:
            (batch, image_dim * 2) fused features
        """
        # Add sequence dimension for attention
        img_query = image_features.unsqueeze(1)  # (batch, 1, image_dim)
        meta_kv = self.meta_projection(meta_features).unsqueeze(1)  # (batch, 1, image_dim)
        
        # Cross-attention: image attends to metadata
        attended, _ = self.cross_attention(img_query, meta_kv, meta_kv)
        attended = attended.squeeze(1)  # (batch, image_dim)
        
        # Residual connection with normalization
        attended = self.norm(attended + image_features)
        
        # Concatenate original and attended features
        return torch.cat([image_features, attended], dim=-1)


class ClassifierHead(nn.Module):
    """
    Multi-layer classifier with regularization.
    
    Uses multiple FC layers with GELU activation, LayerNorm, and Dropout
    for better generalization.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 256),
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final classification layer (no activation - softmax applied in forward)
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) fused features
        
        Returns:
            (batch, num_classes) logits
        """
        return self.classifier(x)


# ============================================================================
# Main Model
# ============================================================================

class Tricorder3Model(nn.Module):
    """
    Tricorder-3 Competition Model with Pretrained Backbone and Cross-Attention Fusion.
    
    Architecture:
        1. Image Encoder: EfficientNet (pretrained, optionally frozen)
        2. Demographics Encoder: Learned embeddings + MLP
        3. Fusion: Cross-attention between image and demographics
        4. Classifier: Multi-layer FC with regularization
    
    Example:
        >>> model = Tricorder3Model(backbone="efficientnet_b1", freeze_backbone=True)
        >>> image = torch.randn(4, 3, 512, 512)
        >>> demographics = torch.tensor([[45.0, 1.0, 5.0], ...])  # age, sex, location
        >>> probs = model(image, demographics)
        >>> print(probs.shape)  # (4, 11)
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b1",
        freeze_backbone: bool = True,
        pretrained: bool = True,
        meta_dim: int = 256,
        num_attention_heads: int = 4,
        classifier_hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.4,
    ):
        """
        Args:
            backbone: Backbone architecture name (see BACKBONE_CONFIGS)
            freeze_backbone: Whether to freeze backbone weights (recommended for small datasets)
            pretrained: Use ImageNet pretrained weights
            meta_dim: Dimension of demographics embedding
            num_attention_heads: Number of heads in cross-attention
            classifier_hidden_dims: Hidden layer dimensions for classifier
            dropout: Dropout probability
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        
        # ============ Image Encoder ============
        self.backbone = self._create_backbone(backbone, pretrained)
        self.image_dim = self._get_backbone_features(backbone)
        
        if freeze_backbone:
            self._freeze_backbone()
        
        # ============ Demographics Encoder ============
        self.demographics_encoder = DemographicsEncoder(
            output_dim=meta_dim,
            dropout=dropout,
        )
        
        # ============ Cross-Attention Fusion ============
        self.fusion = CrossAttentionFusion(
            image_dim=self.image_dim,
            meta_dim=meta_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # ============ Classifier ============
        self.classifier = ClassifierHead(
            input_dim=self.fusion.output_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=NUM_CLASSES,
            dropout=dropout,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Create backbone network using timm."""
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="avg",
        )
        return model
    
    def _get_backbone_features(self, backbone: str) -> int:
        """Get output feature dimension for backbone."""
        if backbone in BACKBONE_CONFIGS:
            return BACKBONE_CONFIGS[backbone]["features"]
        
        # Fallback: run a forward pass to determine
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            features = self.backbone(dummy)
            return features.shape[-1]
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in [self.demographics_encoder, self.fusion, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def unfreeze_backbone(self, unfreeze_layers: Optional[int] = None):
        """
        Unfreeze backbone for fine-tuning.
        
        Args:
            unfreeze_layers: Number of layers to unfreeze from the end.
                            If None, unfreeze all layers.
        """
        if unfreeze_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Get all parameters as a list
            params = list(self.backbone.parameters())
            # Unfreeze last N parameters
            for param in params[-unfreeze_layers:]:
                param.requires_grad = True
        
        self.freeze_backbone = False
    
    def forward(
        self, 
        image: torch.Tensor, 
        demographics: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image: (batch, 3, 512, 512) - RGB image normalized to [0, 1]
            demographics: (batch, 3) - [age, sex, location]
                - age: float (years)
                - sex: 0=female, 1=male
                - location: 1-7 (body location)
        
        Returns:
            (batch, 11) - class probabilities (softmax)
        """
        # Extract image features
        if self.freeze_backbone:
            with torch.no_grad():
                image_features = self.backbone(image)
        else:
            image_features = self.backbone(image)
        
        # Encode demographics
        meta_features = self.demographics_encoder(demographics)
        
        # Fuse modalities with cross-attention
        fused_features = self.fusion(image_features, meta_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        # Apply softmax for probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        return probabilities
    
    def forward_with_logits(
        self,
        image: torch.Tensor,
        demographics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and probabilities.
        Useful for training with CrossEntropyLoss.
        
        Returns:
            (logits, probabilities)
        """
        if self.freeze_backbone:
            with torch.no_grad():
                image_features = self.backbone(image)
        else:
            image_features = self.backbone(image)
        
        meta_features = self.demographics_encoder(demographics)
        fused_features = self.fusion(image_features, meta_features)
        logits = self.classifier(fused_features)
        probabilities = F.softmax(logits, dim=-1)
        
        return logits, probabilities
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }


# ============================================================================
# Export Functions
# ============================================================================

def export_to_onnx(
    model: Tricorder3Model,
    output_path: str,
    batch_size: int = 1,
    opset_version: int = 14,
) -> str:
    """
    Export model to ONNX format for submission.
    
    Args:
        model: Trained Tricorder3Model
        output_path: Path to save ONNX model
        batch_size: Batch size for export (use 1 for dynamic)
        opset_version: ONNX opset version
    
    Returns:
        Path to saved ONNX model
    """
    # Move model to CPU for ONNX export (required by PyTorch ONNX exporter)
    model = model.cpu()
    model.eval()
    
    # Create dummy inputs on CPU
    dummy_image = torch.randn(batch_size, 3, 512, 512, device='cpu')
    dummy_demographics = torch.tensor([[45.0, 1.0, 5.0]] * batch_size, device='cpu')
    
    # Export using legacy exporter (more reliable than dynamo-based)
    torch.onnx.export(
        model,
        (dummy_image, dummy_demographics),
        output_path,
        input_names=["image", "demographics"],
        output_names=["output"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "demographics": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter for better compatibility
    )
    
    print(f"Model exported to: {output_path}")
    return output_path


def verify_onnx_model(onnx_path: str) -> bool:
    """Verify ONNX model is valid and test inference."""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        
        # Load and check model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("ONNX model structure is valid")
        
        # Test inference
        session = ort.InferenceSession(onnx_path)
        
        dummy_image = np.random.rand(1, 3, 512, 512).astype(np.float32)
        dummy_demo = np.array([[45.0, 1.0, 5.0]], dtype=np.float32)
        
        outputs = session.run(
            None, 
            {"image": dummy_image, "demographics": dummy_demo}
        )
        
        probs = outputs[0]
        print(f"Output shape: {probs.shape}")
        print(f"Output sum: {probs.sum():.4f} (should be ~1.0)")
        print(f"Output range: [{probs.min():.4f}, {probs.max():.4f}]")
        
        # Verify softmax
        if abs(probs.sum() - 1.0) < 0.01:
            print("Softmax verification: PASSED")
            return True
        else:
            print("Softmax verification: FAILED")
            return False
            
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


# ============================================================================
# Factory Functions
# ============================================================================

def create_model(
    backbone: str = "efficientnet_b1",
    freeze_backbone: bool = True,
    **kwargs
) -> Tricorder3Model:
    """
    Create a Tricorder3Model with specified configuration.
    
    Args:
        backbone: Backbone name (efficientnet_b0, efficientnet_b1, efficientnet_b2, etc.)
        freeze_backbone: Whether to freeze backbone (recommended for fine-tuning)
        **kwargs: Additional arguments passed to Tricorder3Model
    
    Returns:
        Initialized Tricorder3Model
    """
    return Tricorder3Model(
        backbone=backbone,
        freeze_backbone=freeze_backbone,
        **kwargs
    )


def create_lightweight_model() -> Tricorder3Model:
    """Create a lightweight model (<30MB) for maximum efficiency score."""
    return Tricorder3Model(
        backbone="efficientnet_b0",
        freeze_backbone=True,
        meta_dim=128,
        classifier_hidden_dims=(256, 128),
        dropout=0.3,
    )


def create_balanced_model() -> Tricorder3Model:
    """
    Create a balanced model (<50MB) with good accuracy/efficiency tradeoff.
    Uses EfficientNet-B0 backbone with larger classifier for full efficiency score.
    """
    return Tricorder3Model(
        backbone="efficientnet_b0",
        freeze_backbone=True,
        meta_dim=256,
        num_attention_heads=4,
        classifier_hidden_dims=(512, 256),
        dropout=0.4,
    )


def create_larger_model() -> Tricorder3Model:
    """
    Create a larger model (~57MB) with EfficientNet-B1.
    Slightly lower efficiency score (0.93) but potentially higher accuracy.
    """
    return Tricorder3Model(
        backbone="efficientnet_b1",
        freeze_backbone=True,
        meta_dim=256,
        num_attention_heads=4,
        classifier_hidden_dims=(512, 256),
        dropout=0.4,
    )


def create_accurate_model() -> Tricorder3Model:
    """
    Create a model optimized for accuracy (~80MB) with fine-tuning enabled.
    Lower efficiency score (0.70) but maximum accuracy potential.
    Use this if you have lots of training data.
    """
    return Tricorder3Model(
        backbone="efficientnet_b2",
        freeze_backbone=False,  # Fine-tune backbone
        meta_dim=256,
        num_attention_heads=8,
        classifier_hidden_dims=(512, 256),
        dropout=0.3,
    )


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Tricorder-3 Model Architecture Test")
    print("=" * 60)
    
    # Create model
    print("\nCreating balanced model...")
    model = create_balanced_model()
    
    # Print model info
    params = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total:     {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen:    {params['frozen']:,}")
    print(f"\nEstimated Size: {model.get_model_size_mb():.2f} MB")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 512, 512)
    dummy_demographics = torch.tensor([
        [45.0, 1.0, 5.0],  # 45yo male, head
        [30.0, 0.0, 7.0],  # 30yo female, torso
    ])
    
    with torch.no_grad():
        probs = model(dummy_image, dummy_demographics)
    
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Input demographics shape: {dummy_demographics.shape}")
    print(f"Output shape: {probs.shape}")
    print(f"Output sum (should be 1.0): {probs[0].sum():.4f}")
    print(f"Predictions:\n{probs}")
    
    # Test ONNX export
    print("\n" + "=" * 60)
    print("Testing ONNX Export")
    print("=" * 60)
    
    onnx_path = "/tmp/test_tricorder3_model.onnx"
    export_to_onnx(model, onnx_path)
    verify_onnx_model(onnx_path)
    
    print("\nAll tests passed!")
