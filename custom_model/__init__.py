"""
Custom Model for Tricorder-3 Competition

Usage:
    from custom_model import Tricorder3Model, create_balanced_model, export_to_onnx
    from custom_model import Tricorder3Trainer, TrainingConfig
    
    # Create model
    model = create_balanced_model()
    
    # Train with competition-optimized trainer
    config = TrainingConfig(epochs=50, learning_rate=1e-3)
    trainer = Tricorder3Trainer(model, config)
    trainer.fit(train_dataset, val_dataset)
    
    # Export to ONNX for submission
    export_to_onnx(model, "my_model.onnx")
"""

from .model import (
    # Main model
    Tricorder3Model,
    
    # Advanced model (ISIC competition winner style)
    ISICWinnerModel,
    
    # Components
    DemographicsEncoder,
    CrossAttentionFusion,
    ClassifierHead,
    GeM,
    SEBlock,
    MetadataGating,
    
    # Factory functions
    create_model,
    create_lightweight_model,
    create_balanced_model,
    create_larger_model,
    create_accurate_model,
    
    # Advanced model factory functions
    create_isic_winner_model,
    create_isic_winner_small,
    create_convnext_model,
    
    # Vision Transformer factory functions
    create_vit_model,
    create_vit_small,
    create_deit_model,
    create_deit_small,
    create_swin_model,
    
    # All model creators for easy iteration
    # MODEL_CREATORS,
    
    # Export utilities
    export_to_onnx,
    verify_onnx_model,
    
    # Constants
    NUM_CLASSES,
    IMAGE_SIZE,
    CLASS_NAMES,
    CLASS_WEIGHTS,
    LOCATION_MAP,
    BACKBONE_CONFIGS,
)

__version__ = "1.0.0"
__all__ = [
    # Models
    "Tricorder3Model",
    "ISICWinnerModel",
    # Components
    "DemographicsEncoder",
    "CrossAttentionFusion",
    "ClassifierHead",
    "GeM",
    "SEBlock",
    "MetadataGating",
    # Factory functions
    "create_model",
    "create_lightweight_model",
    "create_balanced_model",
    "create_larger_model",
    "create_accurate_model",
    "create_isic_winner_model",
    "create_isic_winner_small",
    "create_convnext_model",
    "create_vit_model",
    "create_vit_small",
    "create_deit_model",
    "create_deit_small",
    "create_swin_model",
    # Export utilities
    "export_to_onnx",
    "verify_onnx_model",
    # Constants
    "NUM_CLASSES",
    "IMAGE_SIZE",
    "CLASS_NAMES",
    "CLASS_WEIGHTS",
    "LOCATION_MAP",
    "BACKBONE_CONFIGS",
    # Trainer
    "Tricorder3Trainer",
    "TrainingConfig",
    "SkinLesionDataset",
    "CompetitionAlignedLoss",
    "CombinedCompetitionLoss",
    "F1OptimizedLoss",
    "evaluate_predictions",
    "calculate_competition_score",
    "calculate_weighted_f1",
    "get_train_transforms",
    "get_val_transforms",
    "create_weighted_sampler",
    "print_class_distribution",
    "optimize_thresholds",
]

from .trainer import (
    # Trainer
    Tricorder3Trainer,
    TrainingConfig,
    
    # Dataset
    SkinLesionDataset,
    
    # Loss functions
    CompetitionAlignedLoss,
    CombinedCompetitionLoss,
    F1OptimizedLoss,
    
    # Utilities
    evaluate_predictions,
    calculate_competition_score,
    calculate_weighted_f1,
    get_train_transforms,
    get_val_transforms,
    create_weighted_sampler,
    print_class_distribution,
    load_dataset_from_csv,
    optimize_thresholds,  # Post-training threshold optimization
)
