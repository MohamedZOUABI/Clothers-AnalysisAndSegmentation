#!/usr/bin/env python3
"""
Test torchvision Mask R-CNN - Alternative to Detectron2
This approach is often more reliable on Windows
"""

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
import time

def test_torchvision_maskrcnn():
    """Test torchvision Mask R-CNN implementation"""
    print("="*50)
    print("TORCHVISION MASK R-CNN TEST")
    print("="*50)
    
    print(f"Torchvision version: {torchvision.__version__}")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load pre-trained Mask R-CNN model
        print("Loading pre-trained Mask R-CNN model...")
        model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # Modify for DeepFashion2 classes (13 clothing categories + background)
        num_classes = 14  # 13 fashion categories + background
        
        # Replace the classifier head
        from torch import nn
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes)
        
        # Replace the mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes)
        
        print("✓ Model architecture modified for DeepFashion2")
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        print("✓ Model loaded successfully")
        
        # Test with dummy input
        print("Testing inference with dummy image...")
        
        # Create dummy image (3, H, W) format
        dummy_image = torch.randn(3, 480, 640).to(device)
        
        # Measure GPU memory before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            start_time = time.time()
            prediction = model([dummy_image])
            inference_time = time.time() - start_time
        
        print(f"✓ Inference successful! Time: {inference_time:.3f} seconds")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            memory_used = (memory_after - memory_before) / 1024**2
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            available_memory = total_memory - memory_after / 1024**2
            
            print(f"GPU memory used for inference: {memory_used:.1f} MB")
            print(f"Total GPU memory: {total_memory:.1f} MB")
            print(f"Available GPU memory: {available_memory:.1f} MB")
            
            # Training memory estimation (rough)
            estimated_training_memory = memory_used * 3  # Rough estimate
            print(f"Estimated training memory needed: {estimated_training_memory:.1f} MB")
            
            if available_memory < estimated_training_memory:
                print("⚠ WARNING: May need memory optimization for training")
                print("Recommendations:")
                print("- Use batch_size=1")
                print("- Reduce image resolution")
                print("- Use gradient accumulation")
                print("- Consider mixed precision training")
            else:
                print("✓ Should have sufficient memory for training")
        
        # Show prediction structure
        pred = prediction[0]
        print(f"\nPrediction structure:")
        for key, value in pred.items():
            if torch.is_tensor(value):
                print(f"- {key}: shape {value.shape}")
            else:
                print(f"- {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_training_template():
    """Show basic training template"""
    print("\n" + "="*50)
    print("TRAINING TEMPLATE")
    print("="*50)
    
    training_code = '''
# Basic training setup for torchvision Mask R-CNN
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load model
model = maskrcnn_resnet50_fpn(pretrained=True)
num_classes = 14  # DeepFashion2: 13 classes + background

# Modify heads for your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

# Setup for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer (memory-optimized settings)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop would go here...
'''
    print(training_code)

def main():
    """Main test function"""
    print("Testing Torchvision Mask R-CNN for DeepFashion2")
    print("This is a reliable alternative to Detectron2\n")
    
    success = test_torchvision_maskrcnn()
    
    if success:
        show_training_template()
        print("\n" + "="*50)
        print("NEXT STEPS")
        print("="*50)
        print("1. Create DeepFashion2 dataset class")
        print("2. Convert annotations to torchvision format")
        print("3. Set up data loaders with memory optimization")
        print("4. Implement training loop with gradient accumulation")
        print("5. Add validation and checkpointing")
    else:
        print("\n" + "="*50)
        print("TROUBLESHOOTING")
        print("="*50)
        print("If this test failed, try:")
        print("1. Update PyTorch: pip install --upgrade torch torchvision")
        print("2. Clear GPU memory: restart Python kernel")
        print("3. Check CUDA compatibility")

if __name__ == "__main__":
    main()