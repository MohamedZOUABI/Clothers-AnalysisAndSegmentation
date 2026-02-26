#!/usr/bin/env python3
"""
DeepFashion2 Dataset Class for PyTorch/Torchvision Mask R-CNN
Handles DeepFashion2 original format with separate JSON files per image
"""

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image, ImageDraw
import numpy as np
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Any

class DeepFashion2Dataset(Dataset):
    """
    DeepFashion2 dataset for instance segmentation
    
    DeepFashion2 Categories:
    1: short sleeve top, 2: long sleeve top, 3: short sleeve outwear, 4: long sleeve outwear,
    5: vest, 6: sling, 7: shorts, 8: trousers, 9: skirt, 10: short sleeve dress,
    11: long sleeve dress, 12: vest dress, 13: sling dress
    """
    
    CATEGORIES = {
        1: "short_sleeve_top", 2: "long_sleeve_top", 3: "short_sleeve_outwear", 
        4: "long_sleeve_outwear", 5: "vest", 6: "sling", 7: "shorts", 
        8: "trousers", 9: "skirt", 10: "short_sleeve_dress", 
        11: "long_sleeve_dress", 12: "vest_dress", 13: "sling_dress"
    }
    
    def __init__(self, 
                 images_dir: str,
                 annotations_dir: str,
                 transforms=None,
                 image_size: Tuple[int, int] = (640, 480),
                 min_area: int = 100,
                 max_samples: int = None):
        """
        Args:
            images_dir: Directory containing images
            annotations_dir: Directory containing JSON annotation files (one per image)
            transforms: Optional transforms to apply
            image_size: Target image size (width, height)
            min_area: Minimum area for valid annotations
            max_samples: Maximum number of samples to load (for testing)
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_size = image_size
        self.min_area = min_area
        
        print(f"Loading images from: {images_dir}")
        print(f"Loading annotations from: {annotations_dir}")
        
        # Get list of valid image files
        self.image_files = self._get_valid_image_files(max_samples)
        
        print(f"Loaded {len(self.image_files)} images")
    
    def _get_valid_image_files(self, max_samples: int = None) -> List[str]:
        """Get list of valid image files that have corresponding annotations"""
        valid_files = []
        
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory does not exist: {self.images_dir}")
        
        if not os.path.exists(self.annotations_dir):
            raise ValueError(f"Annotations directory does not exist: {self.annotations_dir}")
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(self.images_dir) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        # Check if corresponding annotation exists
        for img_file in image_files:
            # Get annotation filename (replace image extension with .json)
            anno_file = os.path.splitext(img_file)[0] + '.json'
            anno_path = os.path.join(self.annotations_dir, anno_file)
            
            if os.path.exists(anno_path):
                valid_files.append(img_file)
        
        if len(valid_files) == 0:
            raise ValueError("No valid image-annotation pairs found!")
        
        return valid_files
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get item for training
        
        Returns:
            image: Tensor of shape [C, H, W]
            target: Dict with keys 'boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'
        """
        # Get image filename
        img_filename = self.image_files[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Resize image
        image = image.resize(self.image_size, Image.LANCZOS)
        
        # Load annotations
        anno_filename = os.path.splitext(img_filename)[0] + '.json'
        anno_path = os.path.join(self.annotations_dir, anno_filename)
        
        with open(anno_path, 'r') as f:
            annotations = json.load(f)
        
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        
        # Process each item in the annotation
        for item_key, item_data in annotations.items():
            if item_key == 'source' or item_key == 'pair_id':
                continue
            
            if not isinstance(item_data, dict):
                continue
            
            # Get category
            category_id = item_data.get('category_id')
            if category_id is None or category_id not in self.CATEGORIES:
                continue
            
            # Get segmentation
            segmentation = item_data.get('segmentation')
            if not segmentation or len(segmentation) == 0:
                continue
            
            try:
                # Convert segmentation to mask
                mask = self._segmentation_to_mask(segmentation, original_size)
                if mask is None or np.sum(mask) == 0:
                    continue
                
                # Resize mask
                mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize(self.image_size, Image.NEAREST)
                mask = np.array(mask_pil) > 0
                
                # Get bounding box from mask
                bbox = self._mask_to_bbox(mask)
                if bbox is None:
                    continue
                
                # Calculate area
                area = np.sum(mask)
                if area < self.min_area:
                    continue
                
                boxes.append(bbox)
                labels.append(category_id)
                masks.append(mask)
                areas.append(area)
                iscrowd.append(0)
                
            except Exception as e:
                print(f"Error processing {item_key} in {img_filename}: {e}")
                continue
        
        # Convert to tensors
        if len(boxes) == 0:
            # Handle empty annotations - create dummy values
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, self.image_size[1], self.image_size[0]), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        # Convert image to tensor
        if self.transforms:
            image = self.transforms(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, target
    
    def _segmentation_to_mask(self, segmentation: List[List[float]], image_size: Tuple[int, int]) -> np.ndarray:
        """Convert segmentation polygons to binary mask"""
        try:
            mask = Image.new('L', image_size, 0)
            draw = ImageDraw.Draw(mask)
            
            for polygon in segmentation:
                if len(polygon) < 6:
                    continue
                
                # Convert flat list to list of tuples
                polygon_points = []
                for i in range(0, len(polygon), 2):
                    if i + 1 < len(polygon):
                        x, y = polygon[i], polygon[i + 1]
                        # Clamp coordinates to image bounds
                        x = max(0, min(x, image_size[0] - 1))
                        y = max(0, min(y, image_size[1] - 1))
                        polygon_points.append((x, y))
                
                if len(polygon_points) >= 3:
                    draw.polygon(polygon_points, outline=1, fill=1)
            
            return np.array(mask)
        
        except Exception as e:
            print(f"Error converting segmentation to mask: {e}")
            return None
    
    def _mask_to_bbox(self, mask: np.ndarray) -> List[float]:
        """Convert binary mask to bounding box [x1, y1, x2, y2]"""
        try:
            pos = np.where(mask)
            if len(pos[0]) == 0:
                return None
            
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            
            # Ensure valid box
            if xmax <= xmin or ymax <= ymin:
                return None
            
            return [float(xmin), float(ymin), float(xmax), float(ymax)]
        
        except Exception as e:
            print(f"Error converting mask to bbox: {e}")
            return None

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))

def get_transforms(train: bool = True):
    """Get image transforms for training or validation"""
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def test_dataset(images_dir: str, annotations_dir: str, num_samples: int = 5):
    """Test the dataset implementation"""
    print("Testing DeepFashion2 Dataset...")
    print("="*50)
    
    try:
        dataset = DeepFashion2Dataset(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            transforms=None,  # Use None for easier visualization
            image_size=(640, 480),
            max_samples=100  # Only load first 100 for testing
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Test first few samples
        for i in range(min(num_samples, len(dataset))):
            image, target = dataset[i]
            
            print(f"\nSample {i} ({dataset.image_files[i]}):")
            print(f"  Image shape: {image.shape}")
            print(f"  Number of objects: {len(target['boxes'])}")
            print(f"  Labels: {target['labels'].tolist()}")
            print(f"  Box shapes: {target['boxes'].shape}")
            print(f"  Mask shapes: {target['masks'].shape}")
            
            if len(target['boxes']) > 0:
                print(f"  Sample box: {target['boxes'][0].tolist()}")
                print(f"  Sample area: {target['area'][0].item():.1f}")
            
            # Verify data integrity
            assert image.shape[0] == 3, "Image should have 3 channels"
            assert len(target['boxes']) == len(target['labels']), "Boxes and labels mismatch"
            assert len(target['masks']) == len(target['labels']), "Masks and labels mismatch"
        
        print("\n" + "="*50)
        print("✓ Dataset test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("DeepFashion2 Dataset Implementation")
    print("For DeepFashion2 format with separate JSON files per image")