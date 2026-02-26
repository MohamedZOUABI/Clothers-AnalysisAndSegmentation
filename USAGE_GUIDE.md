# DeepFashion2 Model - Usage Guide

## 🎉 Congratulations!

Your model training is complete with excellent results:
- **Final Training Loss**: 0.1614
- **Final Validation Loss**: 0.1579
- **GPU Memory Usage**: ~535 MB (efficient!)
- **Training Time**: ~28.5 minutes per epoch

## 📁 Files You Have

```
checkpoints/
├── best_model.pth          # Your best model (use this!)
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_15.pth
├── checkpoint_epoch_20.pth
└── training_history.json   # Loss curves and metrics
```

---

## 🚀 Quick Start - 3 Ways to Use Your Model

### Method 1: Simple Python Script (Easiest)

```python
from inference_deepfashion2 import DeepFashion2Predictor

# Load model
predictor = DeepFashion2Predictor(
    checkpoint_path='checkpoints/best_model.pth',
    device='cuda',
    confidence_threshold=0.5
)

# Detect clothing in image
results = predictor.get_clothing_info('your_image.jpg')

# Print results
for item in results:
    print(f"{item['category']}: {item['confidence']:.2f}")

# Visualize results
predictor.visualize('your_image.jpg', output_path='result.jpg')
```

**Use this when**: Processing images in a Python script or Jupyter notebook

---

### Method 2: REST API Server (Best for Web Apps)

1. **Install Flask**:
```bash
pip install flask flask-cors
```

2. **Start the server**:
```bash
python api_server.py
```

3. **Use the API** from any language:

**Python Example**:
```python
import requests

# Upload and detect
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/detect',
        files={'image': f},
        data={'confidence': 0.5}
    )
    
results = response.json()
print(results)
```

**JavaScript Example**:
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('confidence', 0.5);

fetch('http://localhost:5000/detect', {
    method: 'POST',
    body: formData
})
.then(r => r.json())
.then(data => console.log(data));
```

**cURL Example**:
```bash
curl -X POST http://localhost:5000/detect \
  -F "image=@your_image.jpg" \
  -F "confidence=0.5"
```

**Use this when**: Building web applications, mobile apps, or integrating with other services

---

### Method 3: Web Interface (Best for Testing)

1. **Start the API server**:
```bash
python api_server.py
```

2. **Open the web interface**:
```bash
# Open web_interface.html in your browser
```

3. **Drag and drop images** to see instant results!

**Use this when**: Testing the model, showing demos, or quick visual inspection

---

## 📊 API Endpoints Reference

### `GET /health`
Check if server is running
```bash
curl http://localhost:5000/health
```

### `POST /detect`
Detect clothing items (returns JSON)
```bash
curl -X POST http://localhost:5000/detect \
  -F "image=@image.jpg" \
  -F "confidence=0.5"
```

**Response**:
```json
{
  "success": true,
  "num_items": 2,
  "items": [
    {
      "category": "trousers",
      "confidence": 0.95,
      "bounding_box": {"x1": 100, "y1": 200, "x2": 300, "y2": 500},
      "mask_area": 45000.0
    }
  ]
}
```

### `POST /detect_visualize`
Returns image with detections drawn
```bash
curl -X POST http://localhost:5000/detect_visualize \
  -F "image=@image.jpg" \
  --output result.jpg
```

### `POST /batch_detect`
Process multiple images at once
```bash
curl -X POST http://localhost:5000/batch_detect \
  -F "images[]=@image1.jpg" \
  -F "images[]=@image2.jpg"
```

### `GET /categories`
Get list of all clothing categories
```bash
curl http://localhost:5000/categories
```

---

## 🎯 Clothing Categories Detected

Your model can detect these 13 clothing types:

1. **short_sleeve_top** - T-shirts, short sleeve shirts
2. **long_sleeve_top** - Long sleeve shirts, blouses
3. **short_sleeve_outwear** - Short sleeve jackets
4. **long_sleeve_outwear** - Coats, long jackets
5. **vest** - Vests, waistcoats
6. **sling** - Tank tops, sleeveless tops
7. **shorts** - Short pants
8. **trousers** - Long pants, jeans
9. **skirt** - All types of skirts
10. **short_sleeve_dress** - Short sleeve dresses
11. **long_sleeve_dress** - Long sleeve dresses
12. **vest_dress** - Sleeveless dresses
13. **sling_dress** - Dress with thin straps

---

## 🔧 Advanced Usage

### Batch Processing Multiple Images

```python
from inference_deepfashion2 import DeepFashion2Predictor
import os
import json

predictor = DeepFashion2Predictor('checkpoints/best_model.pth')

# Process all images in folder
image_folder = 'my_images/'
results_all = {}

for img_file in os.listdir(image_folder):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join(image_folder, img_file)
        results = predictor.get_clothing_info(img_path)
        results_all[img_file] = results
        
        # Save visualization
        predictor.visualize(
            img_path, 
            output_path=f'results/{img_file}',
            show=False
        )

# Save all results
with open('batch_results.json', 'w') as f:
    json.dump(results_all, f, indent=2)
```

### Adjusting Confidence Threshold

```python
# Lower threshold = more detections (may include false positives)
predictor.confidence_threshold = 0.3

# Higher threshold = fewer, more confident detections
predictor.confidence_threshold = 0.7
```

### Custom Visualization Colors

```python
# Modify in inference_deepfashion2.py
predictor.COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    # ... add more colors
]
```

---

## 💡 Integration Examples

### Django Integration

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from inference_deepfashion2 import DeepFashion2Predictor

predictor = DeepFashion2Predictor('path/to/best_model.pth')

@csrf_exempt
def detect_clothing(request):
    if request.method == 'POST':
        image = request.FILES['image']
        # Save temporarily
        temp_path = f'/tmp/{image.name}'
        with open(temp_path, 'wb') as f:
            f.write(image.read())
        
        # Detect
        results = predictor.get_clothing_info(temp_path)
        return JsonResponse({'items': results})
```

### FastAPI Integration

```python
from fastapi import FastAPI, File, UploadFile
from inference_deepfashion2 import DeepFashion2Predictor

app = FastAPI()
predictor = DeepFashion2Predictor('checkpoints/best_model.pth')

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    temp_path = f'/tmp/{file.filename}'
    with open(temp_path, 'wb') as f:
        f.write(contents)
    
    results = predictor.get_clothing_info(temp_path)
    return {"items": results}
```

### Streamlit App

```python
import streamlit as st
from inference_deepfashion2 import DeepFashion2Predictor
from PIL import Image

st.title("Clothing Detection App")

predictor = DeepFashion2Predictor('checkpoints/best_model.pth')

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    
    if st.button('Detect'):
        # Save temp file
        temp_path = 'temp.jpg'
        image.save(temp_path)
        
        # Detect
        results = predictor.get_clothing_info(temp_path)
        
        st.write(f"Found {len(results)} items:")
        for item in results:
            st.write(f"- {item['category']}: {item['confidence']:.2%}")
```

---

## 🎨 Use Cases

### E-commerce
- Automatic product categorization
- Visual search
- Style recommendations

### Fashion Analysis
- Outfit composition analysis
- Trend detection
- Wardrobe management apps

### Retail
- Inventory management
- Virtual try-on systems
- Customer preference analysis

### Content Moderation
- Clothing verification
- Dress code compliance
- Image classification

---

## ⚙️ Performance Tips

### Speed Optimization
```python
# Use smaller confidence threshold for faster processing
predictor.confidence_threshold = 0.7

# Resize large images before processing
from PIL import Image
img = Image.open('large_image.jpg')
img = img.resize((1280, 960))  # Resize to reasonable size
img.save('resized.jpg')
```

### Memory Optimization
```python
# Process one image at a time
# Clear GPU cache between images
import torch

for img_path in image_list:
    results = predictor.predict(img_path)
    # Process results...
    torch.cuda.empty_cache()  # Clear cache
```

### Accuracy Tuning
- **High precision needed**: confidence_threshold = 0.7-0.9
- **High recall needed**: confidence_threshold = 0.3-0.5
- **Balanced**: confidence_threshold = 0.5 (default)

---

## 🐛 Troubleshooting

### Issue: "Out of memory" error
**Solution**: Reduce image size or process on CPU
```python
predictor = DeepFashion2Predictor(
    checkpoint_path='best_model.pth',
    device='cpu'  # Use CPU instead of GPU
)
```

### Issue: Low detection accuracy
**Solutions**:
1. Lower confidence threshold
2. Ensure good image quality
3. Make sure clothing items are clearly visible
4. Train longer or with more data

### Issue: API server not responding
**Check**:
1. Server is running: `python api_server.py`
2. Correct port (5000)
3. Firewall settings
4. Check server logs for errors

---

## 📈 Next Steps

### Improve Your Model
1. **Train longer**: Increase `num_epochs` from 20 to 50
2. **Use more data**: Increase `max_train_samples`
3. **Data augmentation**: Add more transforms
4. **Fine-tune learning rate**: Adjust based on loss curves

### Deploy to Production
1. **Use Docker** for easy deployment
2. **Add authentication** to API
3. **Use HTTPS** for security
4. **Monitor performance** and errors
5. **Set up logging**

### Scale Up
1. **Use batch processing** for multiple images
2. **Add queue system** (Redis, Celery)
3. **Load balancing** for multiple servers
4. **GPU optimization** with TensorRT

---

## 📞 Support

If you have issues:
1. Check this guide
2. Review error messages
3. Test with simple examples first
4. Verify model file exists and is not corrupted

---

## 🎓 Summary

You now have:
- ✅ Trained Mask R-CNN model
- ✅ Inference scripts
- ✅ REST API server
- ✅ Web interface
- ✅ Integration examples

**Start using your model right away with**:
```bash
python quick_inference_example.py
```

**Good luck with your project! 🚀**