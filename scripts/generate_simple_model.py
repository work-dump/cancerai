#!/usr/bin/env python3
"""
Simple Tricorder-3 Model Generator (No Bittensor dependency)

Generates a basic ONNX model for testing the evaluation pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tricorder-3 constants
NUM_CLASSES = 11
NUM_DEMOGRAPHICS = 3  # age, gender, location
IMAGE_SIZE = (512, 512)
FEATURE_CHANNELS = [16, 32, 64]
DEMOGRAPHICS_FEATURES = 16


class SimpleSkinLesionModel(nn.Module):
    def __init__(self):
        super().__init__()
        ch = FEATURE_CHANNELS
        
        # Image feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(ch[0], ch[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(ch[1], ch[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Demographics processor
        self.demographics_processor = nn.Sequential(
            nn.Linear(NUM_DEMOGRAPHICS, DEMOGRAPHICS_FEATURES),
            nn.ReLU(),
            nn.BatchNorm1d(DEMOGRAPHICS_FEATURES)
        )
        
        # Combined classifier
        combined_features = ch[2] + DEMOGRAPHICS_FEATURES  # 64 + 16 = 80
        self.classifier = nn.Linear(combined_features, NUM_CLASSES, bias=False)
    
    def forward(self, image, demographics):
        # Extract image features
        img_features = self.features(image)
        img_features = img_features.view(img_features.size(0), -1)
        
        # Process demographics
        demo_features = self.demographics_processor(demographics)
        
        # Combine and classify
        combined = torch.cat((img_features, demo_features), dim=1)
        logits = self.classifier(combined)
        
        # Apply softmax
        probabilities = F.softmax(logits, dim=1)
        return probabilities


def main():
    import os
    
    output_dir = "/home/ubuntu/Projects/bittensor/cancer-ai/models"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "sample_tricorder_3_model.onnx")
    
    print("Creating Tricorder-3 model...")
    model = SimpleSkinLesionModel()
    model.eval()
    
    # Create dummy inputs
    dummy_image = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    dummy_demo = torch.tensor([[42.0, 1.0, 5.0]])  # age, gender, location
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_image, dummy_demo),
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['image', 'demographics'],
        output_names=['output'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'demographics': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Print model info
    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nModel created successfully!")
    print(f"  Path: {output_path}")
    print(f"  Size: {model_size:.2f} MB")
    print(f"  Input 1: image (batch, 3, 512, 512)")
    print(f"  Input 2: demographics (batch, 3) - [age, gender, location]")
    print(f"  Output: (batch, 11) - class probabilities")


if __name__ == "__main__":
    main()
