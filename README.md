# DeepFashion2 Clothing Detection with Mask R-CNN

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A complete implementation of clothing detection and segmentation using Mask R-CNN on the DeepFashion2 dataset. This project provides instance segmentation, classification, and extraction of clothing items from images.

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Step 1: Verify Environment](#step-1-verify-environment)
  - [Step 2: Test Dataset Loading](#step-2-test-dataset-loading)
  - [Step 3: Train Model](#step-3-train-model)
  - [Step 4: Run Inference](#step-4-run-inference)
  - [Step 5: Extract Clothing Items](#step-5-extract-clothing-items)
- [Model Performance](#model-performance)
- [Clothing Categories](#clothing-categories)
- [API Usage](#api-usage)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## ✨ Features

- **Instance Segmentation**: Detect and segment individual clothing items
- **13 Clothing Categories**: Classify tops, bottoms, dresses, and outerwear
- **Pre-trained on COCO**: Transfer learning from COCO dataset
- **GPU Optimized**: Efficient training on limited VRAM (4GB+)
- **Extract Individual Items**: Save each detected item as separate image
- **JSON Export**: Export detection results in structured JSON format
- **Batch Processing**: Process multiple images efficiently
- **REST API**: Flask-based API for easy integration

## 🖥️ Requirements

### Hardware
- **GPU**: NVIDIA GPU with 4GB+ VRAM (tested on GTX 1050 Ti)
- **RAM**: 8GB+ recommended
- **Storage**: 20GB+ for dataset and models

### Software
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.11
- **CUDA**: 12.1
- **cuDNN**: Compatible with CUDA 12.1

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://gitlab.com/your-username/deepfashion2-maskrcnn.git
cd deepfashion2-maskrcnn
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

**requirements.txt** should contain:
```
torch>=2.5.1
torchvision>=0.20.1
opencv-python>=4.12.0
pillow>=11.3.0
numpy>=2.2.6
matplotlib>=3.10.6
pycocotools>=2.0.10
fvcore>=0.1.5
iopath>=0.1.9
flask>=3.0.0
flask-cors>=4.0.0
```

### 4. Verify Installation

```bash
python Step1-torchvision_maskrcnn_test.py
```

Expected output:
```
✓ Model loaded successfully
✓ Inference successful! Time: 1.138 seconds
✓ Should have sufficient memory for training
```

## 📁 Dataset Preparation

### 1. Download DeepFashion2 Dataset

Download from: [DeepFashion2 Official](https://github.com/switchablenorms/DeepFashion2)

### 2. Organize Dataset Structure

Your `DATA` folder should look like this:

```
DATA/
├── train/
│   ├── image/           # 191,961 training images (000001.jpg - 191961.jpg)
│   └── annos/           # 191,961 annotation files (000001.json - 191961.json)
├── test/
│   └── image/           # 62,629 test images
├── json_for_validation/
│   ├── keypoints_val_information.json
│   ├── retrieval_val_consumer_information.json
│   └── ...
└── json_for_test/
    └── ...
```

### 3. Verify Dataset

```bash
python Step3_test_data_loading.py
```

Expected output:
```
✓ Dataset test passed!
Loaded 1000 images with valid annotations
```

## 📂 Project Structure

```
deepfashion2-maskrcnn/
├── DATA/                              # Dataset folder
│   ├── train/
│   ├── test/
│   └── ...
├── checkpoints/                       # Saved models (created during training)
│   ├── best_model.pth
│   ├── checkpoint_epoch_5.pth
│   └── training_history.json
├── Step1-torchvision_maskrcnn_test.py    # Verify environment
├── Step2_deepfashion2_dataset.py         # Dataset class
├── Step3_test_data_loading.py            # Test dataset loading
├── Step4-train_deepfashion2_maskrcnn.py  # Training script
├── Step5_inference_deepfashion2.py       # Inference class
├── Step6_extract_clothing_items.py       # Extract individual items
├── test_detectron2_installation.py       # Optional: test detectron2
├── nvcc.ipynb                            # Jupyter notebook
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## 🎯 Usage

### Step 1: Verify Environment

Test PyTorch, CUDA, and model loading:

```bash
python Step1-torchvision_maskrcnn_test.py
```

**What it does:**
- Checks PyTorch and CUDA installation
- Tests Mask R-CNN model loading
- Measures GPU memory usage
- Verifies inference capability

### Step 2: Test Dataset Loading

Verify dataset can be loaded correctly:

```bash
python Step3_test_data_loading.py
```

**Configuration:**
Edit paths in the script:
```python
TRAIN_IMAGES = r"C:\path\to\DATA\train\image"
TRAIN_ANNOS = r"C:\path\to\DATA\train\annos"
```

### Step 3: Train Model

Train Mask R-CNN on DeepFashion2:

```bash
python Step4-train_deepfashion2_maskrcnn.py
```

**Training Configuration:**

The script uses optimized settings for 4GB VRAM:

```python
config = {
    'batch_size': 1,                  # Critical for limited VRAM
    'num_epochs': 20,                 # Increase for better results
    'learning_rate': 0.001,
    'max_train_samples': 1000,        # Use subset for initial training
    'max_val_samples': 200,
    'image_size': (640, 480),         # Reduced for memory
}
```

**Training Progress:**
```
Epoch [1/20] Summary:
  Train Loss: 0.8031
  Val Loss: 0.7854
  Learning Rate: 0.001000
  Time: 1709.5s

...

Epoch [20/20] Summary:
  Train Loss: 0.1614
  Val Loss: 0.1579
  ✓ New best model saved!
```

**Training Tips:**
- Start with `max_train_samples=1000` to test
- Increase to 5000-10000 for better performance
- Use full dataset (191,961 images) for production
- Monitor GPU memory with `nvidia-smi`
- Training takes ~8-10 hours for 20 epochs on 1000 images

### Step 4: Run Inference

Detect clothing items in images:

```python
from Step5_inference_deepfashion2 import DeepFashion2Predictor

# Initialize predictor
predictor = DeepFashion2Predictor(
    checkpoint_path='checkpoints/best_model.pth',
    device='cuda',
    confidence_threshold=0.5
)

# Predict on single image
results = predictor.get_clothing_info('test_image.jpg')

# Print results
for item in results:
    print(f"{item['category']}: {item['confidence']:.2%}")

# Visualize results
predictor.visualize('test_image.jpg', output_path='result.jpg')
```

**Command Line Usage:**
```bash
python -c "
from Step5_inference_deepfashion2 import DeepFashion2Predictor
predictor = DeepFashion2Predictor('checkpoints/best_model.pth')
predictor.visualize('your_image.jpg', 'output.jpg')
"
```

### Step 5: Extract Clothing Items

Extract individual clothing items from images:

```bash
python Step6_extract_clothing_items.py
```

**Edit the script to set your image path:**
```python
IMAGE_PATH = r'path\to\your\image.jpg'
```

**Output:**
```
extracted_items/
├── image_results.json              # JSON with all detections
├── image_visualization.jpg         # Full image with boxes
└── items/
    ├── image_item1_long_sleeve_outwear.jpg
    ├── image_item1_long_sleeve_outwear_masked.png
    ├── image_item1_long_sleeve_outwear_mask.png
    ├── image_item2_trousers.jpg
    └── ...
```

**Programmatic Usage:**
```python
from Step6_extract_clothing_items import ClothingExtractor

extractor = ClothingExtractor('checkpoints/best_model.pth')

# Single image
results = extractor.process_image('image.jpg', 'output_folder')

# Batch processing
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
batch_results = extractor.process_batch(images, 'batch_output')
```

## 📊 Model Performance

### Training Results

- **Final Training Loss**: 0.1614
- **Final Validation Loss**: 0.1579
- **Training Time**: ~28 minutes per epoch (GTX 1050 Ti)
- **Inference Time**: ~1.1 seconds per image
- **GPU Memory**: ~535 MB during training

### Performance Metrics

| Metric | Value |
|--------|-------|
| Training Loss | 0.1614 |
| Validation Loss | 0.1579 |
| Inference Speed | 1.1s/image |
| GPU Memory | 535 MB |
| Model Size | 170 MB |

### Loss Curves

Check `checkpoints/training_history.json` for detailed loss progression.

## 👔 Clothing Categories

The model detects and classifies 13 clothing categories:

| ID | Category | Description |
|----|----------|-------------|
| 1 | short_sleeve_top | T-shirts, short sleeve shirts |
| 2 | long_sleeve_top | Long sleeve shirts, blouses |
| 3 | short_sleeve_outwear | Short sleeve jackets |
| 4 | long_sleeve_outwear | Coats, long jackets |
| 5 | vest | Vests, waistcoats |
| 6 | sling | Tank tops, sleeveless tops |
| 7 | shorts | Short pants |
| 8 | trousers | Long pants, jeans |
| 9 | skirt | All types of skirts |
| 10 | short_sleeve_dress | Short sleeve dresses |
| 11 | long_sleeve_dress | Long sleeve dresses |
| 12 | vest_dress | Sleeveless dresses |
| 13 | sling_dress | Dresses with thin straps |

## 🌐 API Usage

### Start API Server

```bash
pip install flask flask-cors
python api_server.py
```

Server runs on `http://localhost:5000`

### API Endpoints

#### `POST /detect`
Detect clothing items (returns JSON)

**Request:**
```bash
curl -X POST http://localhost:5000/detect \
  -F "image=@image.jpg" \
  -F "confidence=0.5"
```

**Response:**
```json
{
  "success": true,
  "num_items": 2,
  "items": [
    {
      "category": "long_sleeve_outwear",
      "confidence": 0.92,
      "bounding_box": {
        "x1": 215, "y1": 343,
        "x2": 858, "y2": 1204
      }
    }
  ]
}
```

#### `POST /detect_visualize`
Returns image with detections

```bash
curl -X POST http://localhost:5000/detect_visualize \
  -F "image=@image.jpg" \
  --output result.jpg
```

#### `GET /health`
Health check

```bash
curl http://localhost:5000/health
```

#### `GET /categories`
List all categories

```bash
curl http://localhost:5000/categories
```

## 🔧 Troubleshooting

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size in training config (already set to 1)
2. Reduce image size: `image_size=(480, 320)`
3. Use CPU: `device='cpu'` (slower but works)
4. Close other GPU applications

### Model Not Loading

**Error:** `FileNotFoundError: checkpoints/best_model.pth`

**Solution:** Train model first with Step 4, or download pre-trained weights

### Low Detection Accuracy

**Solutions:**
1. Lower confidence threshold: `confidence_threshold=0.3`
2. Train with more data: increase `max_train_samples`
3. Train for more epochs: `num_epochs=50`
4. Use higher resolution: `image_size=(800, 600)`

### Dataset Loading Error

**Error:** `Permission denied` or `No valid image-annotation pairs found`

**Solutions:**
1. Check folder paths in scripts
2. Verify dataset structure matches expected format
3. Ensure annotation JSON files exist for each image

### Slow Training

**Solutions:**
1. Use GPU (not CPU)
2. Increase batch size if you have more VRAM
3. Reduce image resolution
4. Use fewer training samples for testing

## 📚 Additional Resources

### DeepFashion2 Dataset
- [Official Repository](https://github.com/switchablenorms/DeepFashion2)
- [Paper](https://arxiv.org/abs/1901.07973)

### PyTorch & Torchvision
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)

### Mask R-CNN
- [Original Paper](https://arxiv.org/abs/1703.06870)
- [Detectron2](https://github.com/facebookresearch/detectron2)

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{deepfashion2,
  title={DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images},
  author={Yuying Ge and Ruimao Zhang and Lingyun Wu and Xiaogang Wang and Xiaoou Tang and Ping Luo},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{maskrcnn,
  title={Mask R-CNN},
  author={He, Kaiming and Gkioxari, Georgia and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={ICCV},
  year={2017}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 🙋 Support

For questions or issues:
- Open an issue on GitLab
- Check the [Troubleshooting](#troubleshooting) section
- Review training logs in `checkpoints/training_history.json`

## 🎯 Future Improvements

- [ ] Add data augmentation for better accuracy
- [ ] Implement color detection for clothing items
- [ ] Add pattern/texture classification
- [ ] Support for video input
- [ ] Mobile deployment (TensorFlow Lite, ONNX)
- [ ] Web interface with drag-and-drop
- [ ] Docker containerization
- [ ] Automated testing suite

## 📝 Changelog

### Version 1.0.0 (2024-10-09)
- Initial release
- Mask R-CNN training pipeline
- Inference and visualization
- Individual item extraction
- REST API server
- Complete documentation

---

**Made with ❤️ for Fashion AI Applications**