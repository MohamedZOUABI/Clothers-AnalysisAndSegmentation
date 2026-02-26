#!/usr/bin/env python3
"""
Setup script to verify environment and prepare directories
Run this after cloning the repository
"""

import os
import sys
import subprocess
import torch

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python_version():
    """Check Python version"""
    print_section("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False

def check_pytorch():
    """Check PyTorch installation"""
    print_section("Checking PyTorch Installation")
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print("✓ PyTorch with CUDA is ready")
        else:
            print("⚠ CUDA not available (CPU mode only)")
        
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

def check_dependencies():
    """Check other dependencies"""
    print_section("Checking Dependencies")
    
    required = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print_section("Creating Directories")
    
    directories = [
        'checkpoints',
        'results',
        'extracted_items',
        'DATA',
        'DATA/train',
        'DATA/train/image',
        'DATA/train/annos',
        'DATA/test',
        'DATA/test/image',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created/verified: {directory}/")
    
    # Create .gitkeep files
    gitkeep_dirs = [
        'checkpoints',
        'DATA',
        'DATA/train',
        'DATA/train/image',
        'DATA/train/annos',
        'DATA/test',
        'DATA/test/image',
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            open(gitkeep_path, 'w').close()
    
    print("\n✓ All directories created")
    return True

def check_dataset():
    """Check if dataset exists"""
    print_section("Checking Dataset")
    
    train_images = 'DATA/train/image'
    train_annos = 'DATA/train/annos'
    
    if os.path.exists(train_images) and os.path.exists(train_annos):
        num_images = len([f for f in os.listdir(train_images) if f.endswith('.jpg')])
        num_annos = len([f for f in os.listdir(train_annos) if f.endswith('.json')])
        
        if num_images > 0 and num_annos > 0:
            print(f"✓ Found {num_images} training images")
            print(f"✓ Found {num_annos} annotation files")
            return True
        else:
            print("⚠ Dataset folders exist but are empty")
            print("\nPlease download DeepFashion2 dataset:")
            print("https://github.com/switchablenorms/DeepFashion2")
            return False
    else:
        print("⚠ Dataset not found")
        print("\nPlease download and extract DeepFashion2 dataset to DATA/ folder")
        print("https://github.com/switchablenorms/DeepFashion2")
        return False

def print_next_steps():
    """Print next steps"""
    print_section("Next Steps")
    
    print("""
1. If dataset is not downloaded:
   - Download DeepFashion2 from: https://github.com/switchablenorms/DeepFashion2
   - Extract to DATA/ folder

2. Verify environment:
   python Step1-torchvision_maskrcnn_test.py

3. Test dataset loading:
   python Step3_test_data_loading.py

4. Start training:
   python Step4-train_deepfashion2_maskrcnn.py

5. Run inference:
   python Step5_inference_deepfashion2.py

6. Extract clothing items:
   python Step6_extract_clothing_items.py

For detailed instructions, see README.md
""")

def main():
    """Main setup function"""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   DeepFashion2 Mask R-CNN Setup                        ║
    ║   Verifying environment and preparing directories      ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    checks = {
        'Python Version': check_python_version(),
        'PyTorch': check_pytorch(),
        'Dependencies': check_dependencies(),
        'Directories': create_directories(),
        'Dataset': check_dataset(),
    }
    
    print_section("Setup Summary")
    
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n✓ Setup completed successfully!")
        print("You can now start training the model.")
    else:
        print("\n⚠ Some checks failed. Please resolve issues before training.")
    
    print_next_steps()
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)