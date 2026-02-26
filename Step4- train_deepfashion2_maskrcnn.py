#!/usr/bin/env python3
"""
Training script for DeepFashion2 with Torchvision Mask R-CNN
Optimized for GTX 1050 Ti (4GB VRAM)
"""
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import os
import time
import json
from datetime import datetime
from Step2_deepfashion2_dataset import DeepFashion2Dataset, collate_fn, get_transforms
import numpy as np

def get_model(num_classes: int, device: torch.device):
    """
    Load Mask R-CNN model with pre-trained COCO weights
    Modified for DeepFashion2 classes
    """
    # Load pre-trained model
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")  # Updated syntax
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    model.to(device)
    return model

def train_one_epoch(model, data_loader, optimizer, device, epoch, print_freq=10):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    num_batches = 0
    
    for i, (images, targets) in enumerate(data_loader):
        try:
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v 
                       for k, v in t.items()} for t in targets]
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            optimizer.step()
            
            # Statistics
            running_loss += losses.item()
            num_batches += 1
            
            if i % print_freq == 0:
                print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], "
                      f"Loss: {losses.item():.4f}")
                
                # Print individual losses
                loss_str = ", ".join([f"{k}: {v.item():.4f}" 
                                     for k, v in loss_dict.items()])
                print(f"  Individual losses: {loss_str}")
                
                # GPU memory info
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**2
                    print(f"  GPU Memory: {memory_used:.1f} MB")
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: Out of memory at step {i}. Skipping batch.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def validate_model(model, data_loader, device):
    """Validate the model"""
    # model.eval() # this cause : AttributeError: 'list' object has no attribute 'values' 
    # -> so i change it to : 
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            try:
                # Move to device
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) if torch.is_tensor(v) else v 
                           for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                total_loss += losses.item()
                num_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: Out of memory during validation. Skipping batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def main():
    """Main training function"""
    # Configuration
    config = {
        # Paths - UPDATED FOR YOUR DATA STRUCTURE
        'train_images_dir': r'C:\Users\mouhamed\Desktop\VESTIFY PROJECT\CODE\Clothers Image Analysis\DeepFashion2\DATA\train\image',
        'train_annotations_dir': r'C:\Users\mouhamed\Desktop\VESTIFY PROJECT\CODE\Clothers Image Analysis\DeepFashion2\DATA\train\annos',
        'val_images_dir': r'C:\Users\mouhamed\Desktop\VESTIFY PROJECT\CODE\Clothers Image Analysis\DeepFashion2\DATA\test\image',
        'val_annotations_dir': r'C:\Users\mouhamed\Desktop\VESTIFY PROJECT\CODE\Clothers Image Analysis\DeepFashion2\DATA\test\image',  # Test set doesn't have annos - use train for now
        
        # Model parameters
        'num_classes': 14,  # 13 DeepFashion2 classes + background
        'image_size': (640, 480),  # (width, height) - reduced for memory
        
        # Training parameters - OPTIMIZED FOR GTX 1050 Ti
        'batch_size': 1,
        'num_epochs': 20,
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'lr_step_size': 7,
        'lr_gamma': 0.1,
        
        # Data loading
        'num_workers': 2,
        'pin_memory': True,
        
        # Dataset limits (for testing/memory)
        'max_train_samples': 1000,  # Use subset for initial training
        'max_val_samples': 200,     # Use subset for validation
        
        # Checkpointing
        'save_every': 5,
        'output_dir': 'checkpoints',
    }
    
    print("DeepFashion2 Mask R-CNN Training")
    print("="*50)
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Datasets
    print("\nLoading datasets...")
    
    try:
        train_dataset = DeepFashion2Dataset(
            images_dir=config['train_images_dir'],
            annotations_dir=config['train_annotations_dir'],
            transforms=get_transforms(train=True),
            image_size=config['image_size'],
            max_samples=config.get('max_train_samples')
        )
        
        # For validation, we'll use a subset of training data since test set has no annotations
        val_dataset = DeepFashion2Dataset(
            images_dir=config['train_images_dir'],
            annotations_dir=config['train_annotations_dir'],
            transforms=get_transforms(train=False),
            image_size=config['image_size'],
            max_samples=config.get('max_val_samples')
        )
        
        print(f"Train dataset: {len(train_dataset)} images")
        print(f"Validation dataset: {len(val_dataset)} images")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please update the paths in the config section")
        return
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['pin_memory']
    )
    
    # Model
    print("\nLoading model...")
    model = get_model(config['num_classes'], device)
    
    # Optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )
    
    # Training loop
    print("\nStarting training...")
    print("="*50)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch + 1
        )
        
        # Validate
        val_loss = validate_model(model, val_loader, device)
        
        # Learning rate step
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Time tracking
        epoch_time = time.time() - epoch_start_time
        
        # Log results
        print(f"\nEpoch [{epoch+1}/{config['num_epochs']}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'time': epoch_time
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(config['output_dir'], 'best_model.pth'))
            print(f"  New best model saved! Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  Checkpoint saved: epoch_{epoch+1}.pth")
        
        # Save training history
        with open(os.path.join(config['output_dir'], 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print("-" * 50)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {config['output_dir']}")



    # === Plot training results ===
    epochs = [h['epoch'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    val_losses = [h['val_loss'] for h in training_history]
    lrs = [h['lr'] for h in training_history]

    plt.figure(figsize=(10, 6))

    # Loss curves
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='s', label='Val Loss')

    # Labels
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DeepFashion2 Mask R-CNN Training Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Learning rate on second axis
    ax2 = plt.gca().twinx()
    ax2.plot(epochs, lrs, color='red', linestyle='--', label='Learning Rate')
    ax2.set_ylabel("Learning Rate")
    ax2.legend(loc="upper right")

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], "training_curve.png"))
    plt.show()



if __name__ == "__main__":
    main()