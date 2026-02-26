#!/usr/bin/env python3
"""
Extract individual clothing items from images and save results as JSON
Creates cropped images for each detected item
"""

from Step5_inference_deepfashion2 import DeepFashion2Predictor
import cv2
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image

class ClothingExtractor:
    """Extract and save individual clothing items from images"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', confidence_threshold: float = 0.5):
        """Initialize the extractor with trained model"""
        self.predictor = DeepFashion2Predictor(
            checkpoint_path=checkpoint_path,
            device=device,
            confidence_threshold=confidence_threshold
        )
    
    def process_image(self, image_path: str, output_dir: str = 'extracted_items'):
        """
        Process image and extract individual clothing items
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save extracted items
            
        Returns:
            Dictionary with results and paths to extracted items
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        items_dir = os.path.join(output_dir, 'items')
        os.makedirs(items_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"\nProcessing: {image_path}")
        print("="*60)
        
        # Get predictions
        predictions = self.predictor.predict(image_path)
        
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare results
        results = {
            'image_path': image_path,
            'timestamp': timestamp,
            'total_items': len(predictions['boxes']),
            'items': []
        }
        
        print(f"\n✓ Found {len(predictions['boxes'])} clothing items:\n")
        
        # Extract each item
        for i in range(len(predictions['boxes'])):
            box = predictions['boxes'][i].astype(int)
            label = predictions['labels'][i]
            score = predictions['scores'][i]
            mask = predictions['masks'][i, 0]
            category = predictions['category_names'][i]
            
            # Print item info
            print(f"  {i+1}. {category.replace('_', ' ').title()}: {score:.2%}")
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box
            
            # Add padding (10 pixels)
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # Crop the item
            cropped_item = image_rgb[y1:y2, x1:x2].copy()
            
            # Crop the mask
            mask_binary = (mask > 0.5).astype(np.uint8)
            cropped_mask = mask_binary[y1:y2, x1:x2]
            
            # Apply mask to create transparent background (optional)
            # Create RGBA image
            cropped_item_rgba = np.zeros((cropped_item.shape[0], cropped_item.shape[1], 4), dtype=np.uint8)
            cropped_item_rgba[:, :, :3] = cropped_item
            cropped_item_rgba[:, :, 3] = cropped_mask * 255  # Alpha channel
            
            # Save cropped item (with background)
            item_filename = f"{base_name}_item{i+1}_{category}.jpg"
            item_path = os.path.join(items_dir, item_filename)
            cropped_bgr = cv2.cvtColor(cropped_item, cv2.COLOR_RGB2BGR)
            cv2.imwrite(item_path, cropped_bgr)
            
            # Save cropped item with transparent background
            item_filename_png = f"{base_name}_item{i+1}_{category}_masked.png"
            item_path_png = os.path.join(items_dir, item_filename_png)
            cropped_pil = Image.fromarray(cropped_item_rgba, 'RGBA')
            cropped_pil.save(item_path_png)
            
            # Save mask separately
            mask_filename = f"{base_name}_item{i+1}_{category}_mask.png"
            mask_path = os.path.join(items_dir, mask_filename)
            mask_image = (cropped_mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_image)
            
            # Add to results
            item_data = {
                'item_id': i + 1,
                'category': category,
                'category_readable': category.replace('_', ' ').title(),
                'confidence': float(score),
                'bounding_box': {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                },
                'mask_area': float(np.sum(mask_binary)),
                'files': {
                    'cropped_image': item_path,
                    'masked_image': item_path_png,
                    'mask': mask_path
                }
            }
            results['items'].append(item_data)
        
        # Save full visualization
        viz_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
        self.predictor.visualize(image_path, output_path=viz_path, show=False)
        results['visualization'] = viz_path
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"{base_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        results['json_file'] = json_path
        
        print(f"\n{'='*60}")
        print("✓ Extraction complete!")
        print(f"  - Extracted {len(results['items'])} items")
        print(f"  - Saved to: {output_dir}/")
        print(f"  - JSON results: {json_path}")
        print(f"  - Visualization: {viz_path}")
        print(f"{'='*60}\n")
        
        return results
    
    def process_batch(self, image_paths: list, output_dir: str = 'extracted_items'):
        """Process multiple images"""
        all_results = []
        
        for image_path in image_paths:
            try:
                result = self.process_image(image_path, output_dir)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                all_results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        # Save batch results
        batch_json_path = os.path.join(output_dir, 'batch_results.json')
        with open(batch_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nBatch processing complete!")
        print(f"Results saved to: {batch_json_path}")
        
        return all_results

def main():
    """Example usage"""
    print("="*60)
    print("Clothing Item Extractor")
    print("="*60)
    
    # Configuration
    CHECKPOINT_PATH = 'checkpoints/best_model.pth'
    IMAGE_PATH = r'C:\Users\mouhamed\Desktop\VESTIFY PROJECT\CODE\Clothers Image Analysis\DeepFashion2\results\visualization.jpg'  # Update this
    OUTPUT_DIR = 'extracted_items'
    CONFIDENCE_THRESHOLD = 0.5
    
    # Initialize extractor
    extractor = ClothingExtractor(
        checkpoint_path=CHECKPOINT_PATH,
        device='cuda',
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    # Process single image
    results = extractor.process_image(IMAGE_PATH, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total items detected: {results['total_items']}\n")
    
    for item in results['items']:
        print(f"Item {item['item_id']}: {item['category_readable']}")
        print(f"  Confidence: {item['confidence']:.1%}")
        print(f"  Size: {item['bounding_box']['width']}x{item['bounding_box']['height']} pixels")
        print(f"  Files:")
        print(f"    - Image: {os.path.basename(item['files']['cropped_image'])}")
        print(f"    - Masked: {os.path.basename(item['files']['masked_image'])}")
        print(f"    - Mask: {os.path.basename(item['files']['mask'])}")
        print()
    
    print("="*60)
    print(f"All files saved in: {OUTPUT_DIR}/")
    print("="*60)
    
    # Example: Process multiple images
    # image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # batch_results = extractor.process_batch(image_list, OUTPUT_DIR)

if __name__ == "__main__":
    main()