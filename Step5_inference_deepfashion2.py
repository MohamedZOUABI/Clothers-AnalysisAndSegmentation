#!/usr/bin/env python3
"""
Inference script for trained DeepFashion2 Mask R-CNN model
Use this to make predictions on real images in your project
"""

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from typing import List, Dict, Tuple

class DeepFashion2Predictor:
    """
    Predictor class for DeepFashion2 trained model
    """
    
    CATEGORIES = {
        1: "short_sleeve_top", 2: "long_sleeve_top", 3: "short_sleeve_outwear", 
        4: "long_sleeve_outwear", 5: "vest", 6: "sling", 7: "shorts", 
        8: "trousers", 9: "skirt", 10: "short_sleeve_dress", 
        11: "long_sleeve_dress", 12: "vest_dress", 13: "sling_dress"
    }
    
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0)
    ]
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', confidence_threshold: float = 0.5):
        """
        Initialize predictor with trained model
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            device: 'cuda' or 'cpu'
            confidence_threshold: Minimum confidence score for detections
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        print(f"Loading model from: {checkpoint_path}")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def _load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint"""
        # Create model architecture
        model = maskrcnn_resnet50_fpn(weights=None)
        
        num_classes = 14  # 13 fashion categories + background
        
        # Modify heads
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        return model
    
    def predict(self, image_path: str) -> Dict:
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with predictions containing:
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - labels: List of category IDs
                - scores: List of confidence scores
                - masks: List of segmentation masks
                - category_names: List of category names
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Convert to tensor
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]
        
        # Filter by confidence threshold
        keep_indices = predictions['scores'] >= self.confidence_threshold
        
        results = {
            'boxes': predictions['boxes'][keep_indices].cpu().numpy(),
            'labels': predictions['labels'][keep_indices].cpu().numpy(),
            'scores': predictions['scores'][keep_indices].cpu().numpy(),
            'masks': predictions['masks'][keep_indices].cpu().numpy(),
            'category_names': [self.CATEGORIES.get(int(label), 'unknown') 
                              for label in predictions['labels'][keep_indices].cpu().numpy()],
            'image_size': original_size
        }
        
        return results
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Make predictions on multiple images"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(None)
        return results
    
    def visualize(self, image_path: str, output_path: str = None, show: bool = True):
        """
        Visualize predictions on image
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            show: Whether to display the image
        """
        # Get predictions
        predictions = self.predict(image_path)
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw predictions
        image_with_predictions = self._draw_predictions(image, predictions)
        
        # Show
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_with_predictions)
            plt.axis('off')
            plt.title(f'Detected {len(predictions["boxes"])} clothing items')
            plt.tight_layout()
            plt.show()
        
        # Save
        if output_path:
            result_bgr = cv2.cvtColor(image_with_predictions, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            print(f"Saved visualization to: {output_path}")
        
        return image_with_predictions
    
    def _draw_predictions(self, image: np.ndarray, predictions: Dict) -> np.ndarray:
        """Draw bounding boxes, masks, and labels on image"""
        image_draw = image.copy()
        
        # Draw each detection
        for i in range(len(predictions['boxes'])):
            box = predictions['boxes'][i].astype(int)
            label_id = predictions['labels'][i]
            score = predictions['scores'][i]
            mask = predictions['masks'][i, 0]
            category_name = predictions['category_names'][i]
            
            # Get color for this category
            color = self.COLORS[label_id % len(self.COLORS)]
            
            # Draw mask
            mask_binary = (mask > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(image_draw)
            colored_mask[mask_binary == 1] = color
            image_draw = cv2.addWeighted(image_draw, 1.0, colored_mask, 0.4, 0)
            
            # Draw bounding box
            cv2.rectangle(image_draw, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Draw label
            label_text = f'{category_name}: {score:.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                image_draw,
                (box[0], box[1] - text_height - 10),
                (box[0] + text_width, box[1]),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                image_draw,
                label_text,
                (box[0], box[1] - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return image_draw
    
    def get_clothing_info(self, image_path: str) -> List[Dict]:
        """
        Extract structured clothing information from image
        
        Returns:
            List of dictionaries with clothing item details
        """
        predictions = self.predict(image_path)
        
        clothing_items = []
        for i in range(len(predictions['boxes'])):
            item = {
                'category': predictions['category_names'][i],
                'confidence': float(predictions['scores'][i]),
                'bounding_box': {
                    'x1': int(predictions['boxes'][i][0]),
                    'y1': int(predictions['boxes'][i][1]),
                    'x2': int(predictions['boxes'][i][2]),
                    'y2': int(predictions['boxes'][i][3])
                },
                'mask_area': float(np.sum(predictions['masks'][i] > 0.5))
            }
            clothing_items.append(item)
        
        return clothing_items

def main():
    """Example usage"""
    print("DeepFashion2 Inference Script")
    print("="*50)
    
    # Configuration
    CHECKPOINT_PATH = 'checkpoints/best_model.pth'
    # TEST_IMAGE = r"C:\Users\mouhamed\Desktop\VESTIFY PROJECT\CODE\Clothers Image Analysis\img\white teeshirt.jpg" 
    TEST_IMAGE = r"C:\Users\mouhamed\Desktop\VESTIFY PROJECT\CODE\Clothers Image Analysis\img\product-media.jpg" 
    OUTPUT_DIR = 'results'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize predictor
    predictor = DeepFashion2Predictor(
        checkpoint_path=CHECKPOINT_PATH,
        device='cuda',
        confidence_threshold=0.5
    )
    
    # Example 1: Simple prediction
    print("\n1. Making prediction on single image...")
    predictions = predictor.predict(TEST_IMAGE)
    print(f"Found {len(predictions['boxes'])} clothing items")
    for i, (cat, score) in enumerate(zip(predictions['category_names'], predictions['scores'])):
        print(f"  {i+1}. {cat}: {score:.2f}")
    
    # Example 2: Visualize results
    print("\n2. Visualizing predictions...")
    output_path = os.path.join(OUTPUT_DIR, 'visualization.jpg')
    predictor.visualize(TEST_IMAGE, output_path=output_path, show=False)
    
    # Example 3: Get structured clothing info
    print("\n3. Getting structured clothing information...")
    clothing_info = predictor.get_clothing_info(TEST_IMAGE)
    print(f"Clothing items found: {len(clothing_info)}")
    for item in clothing_info:
        print(f"  - {item['category']} (confidence: {item['confidence']:.2f})")
    
    # Example 4: Batch prediction
    print("\n4. Batch prediction example...")
    image_list = [TEST_IMAGE]  # Add more images here
    batch_results = predictor.predict_batch(image_list)
    print(f"Processed {len(batch_results)} images")
    
    print("\n" + "="*50)
    print("Inference completed!")
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()