#!/usr/bin/env python3
"""
Test script to verify Detectron2 installation and GPU compatibility
"""

import torch
import sys

def test_pytorch():
    """Test PyTorch installation and CUDA"""
    print("="*50)
    print("PYTORCH TEST")
    print("="*50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test GPU memory allocation
        try:
            x = torch.randn(1000, 1000).cuda()
            print("✓ GPU tensor allocation successful")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU tensor allocation failed: {e}")
    else:
        print("✗ CUDA not available")

def test_detectron2():
    """Test Detectron2 installation"""
    print("\n" + "="*50)
    print("DETECTRON2 TEST")
    print("="*50)
    
    try:
        import detectron2
        print(f"✓ Detectron2 version: {detectron2.__version__}")
        
        # Test model loading
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        
        print("✓ Detectron2 imports successful")
        
        # Test config loading
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("✓ Config setup successful")
        
        # Test model loading (this will download weights if not cached)
        print("Downloading pre-trained weights (this may take a while)...")
        predictor = DefaultPredictor(cfg)
        print("✓ Model loading successful")
        
        # Test memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
            
            # Create a dummy input
            import numpy as np
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            try:
                outputs = predictor(dummy_image)
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / 1024**2
                print(f"✓ Inference successful. GPU memory used: {memory_used:.1f} MB")
                
                # Check if we have enough memory for training
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
                available_memory = total_memory - memory_after / 1024**2
                print(f"Available GPU memory after model loading: {available_memory:.1f} MB")
                
                if available_memory < 1000:  # Less than 1GB
                    print("⚠ WARNING: Low GPU memory. Training may require optimization.")
                else:
                    print("✓ Sufficient GPU memory for training")
                    
            except Exception as e:
                print(f"✗ Inference failed: {e}")
        
        return True
        
    except ImportError:
        print("✗ Detectron2 not installed")
        return False
    except Exception as e:
        print(f"✗ Detectron2 test failed: {e}")
        return False

def test_mmdetection():
    """Test MMDetection as alternative"""
    print("\n" + "="*50)
    print("MMDETECTION TEST (Alternative)")
    print("="*50)
    
    try:
        import mmdet
        print(f"✓ MMDetection version: {mmdet.__version__}")
        
        from mmdet.apis import init_detector
        import mmcv
        print("✓ MMDetection imports successful")
        
        return True
        
    except ImportError:
        print("✗ MMDetection not installed")
        return False
    except Exception as e:
        print(f"✗ MMDetection test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Testing Deep Learning Environment for DeepFashion2 Training")
    
    # Test PyTorch
    test_pytorch()
    
    # Test Detectron2
    detectron2_ok = test_detectron2()
    
    # Test MMDetection as alternative
    if not detectron2_ok:
        mmdet_ok = test_mmdetection()
        if not mmdet_ok:
            print("\n" + "="*50)
            print("INSTALLATION RECOMMENDATIONS")
            print("="*50)
            print("Neither Detectron2 nor MMDetection is installed.")
            print("Please try one of these installation methods:")
            print()
            print("For Detectron2:")
            print("pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.5/index.html")
            print()
            print("For MMDetection:")
            print("pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.5/index.html")
            print("pip install mmdet")
    
    print("\n" + "="*50)
    print("TEST COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()