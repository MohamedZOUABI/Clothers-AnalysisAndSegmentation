#!/usr/bin/env python3
"""
Test script to verify DeepFashion2 data loading before training
"""

from Step2_deepfashion2_dataset import test_dataset

# Test your data
print("Testing DeepFashion2 Data Loading")
print("="*50)

# Update these paths to match your data
TRAIN_IMAGES = r"C:\Users\mouhamed\Desktop\VESTIFY PROJECT\CODE\Clothers Image Analysis\DeepFashion2\DATA\train\image"
TRAIN_ANNOS = r"C:\Users\mouhamed\Desktop\VESTIFY PROJECT\CODE\Clothers Image Analysis\DeepFashion2\DATA\train\annos"

print(f"Images directory: {TRAIN_IMAGES}")
print(f"Annotations directory: {TRAIN_ANNOS}")
print()

# Run test
success = test_dataset(
    images_dir=TRAIN_IMAGES,
    annotations_dir=TRAIN_ANNOS,
    num_samples=5
)

if success:
    print("\n" + "="*50)
    print("✓ Data loading successful!")
    print("You can now run the training script:")
    print("python train_deepfashion2_maskrcnn.py")
else:
    print("\n" + "="*50)
    print("✗ Data loading failed. Please check the error messages above.")