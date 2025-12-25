# **Animal Image Segmentation with UNet**

## **📌 Project Overview**
This project implements a semantic segmentation system for animal images using UNet architecture. The pipeline includes data collection, annotation, mask generation, and model training to segment animals from their backgrounds.

## **🚀 Features**
- Complete pipeline from data collection to model deployment
- Custom dataset creation with animal images
- Binary mask generation from COCO JSON annotations
- UNet-based semantic segmentation model
- Data augmentation techniques for improved generalization

## **📁 Project Structure**
```
animal-segmentation-unet/
├── dataset/
│   ├── images/                    # Original animal images
│   │   ├── train/                 # Training images
│   │   └── val/                   # Validation images
│   └── masks/                     # Binary segmentation masks
│       ├── train/                 # Training masks
│       └── val/                   # Validation masks
├── annotations/
│   └── _annotations.coco.json     # COCO format annotations
├── models/
│   └── unet_segmentation_model.h5 # Trained UNet model
├── src/
│   ├── 01_data_collection.py      # Data gathering script
│   ├── 02_annotation_prep.py      # Annotation preparation
│   ├── 03_mask_generation.py      # Binary mask creation
│   ├── 04_unet_model.py           # UNet model implementation
│   └── 05_training.py             # Training pipeline
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_evaluation.ipynb
├── requirements.txt
├── config.yaml
└── README.md
```

## **🛠️ Installation**

### **Prerequisites**
- Python 3.8+
- CUDA-capable GPU (recommended for training)

### **Setup**
```bash
# Clone the repository
git clone https://github.com/yourusername/animal-segmentation-unet.git
cd animal-segmentation-unet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## **📦 Dependencies**
```txt
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
albumentations>=1.2.0
scikit-learn>=1.0.0
tqdm>=4.64.0
pycocotools>=2.0.0
```

## **📊 Dataset Preparation**

### **1. Data Collection**
- Collected 500+ diverse animal images from open-source repositories
- Images include various species, poses, and backgrounds
- All images are royalty-free and suitable for research

### **2. Annotation Process**
- Used CVAT (Computer Vision Annotation Tool) for manual annotation
- Applied polygon annotations around animal boundaries
- Implemented data augmentation within CVAT:
  - Random rotation (±15°)
  - Horizontal/Vertical flips
  - Brightness/Contrast adjustments
  - Scale variations

### **3. Mask Generation**
Converted COCO JSON annotations to binary masks:
```python
python src/03_mask_generation.py
```
**Output**: Binary masks (0=background, 255=animal) for all annotated images

## **🤖 Model Architecture**

### **UNet Configuration**
- **Input**: 256×256×3 RGB images
- **Encoder**: 4 downsampling blocks with increasing filters (64→128→256→512)
- **Bottleneck**: 1024 filters
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: 256×256×1 binary segmentation mask
- **Activation**: Sigmoid for binary classification

### **Training Parameters**
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 8
- **Epochs**: 50 (with early stopping)
- **Metrics**: Accuracy, IoU (Intersection over Union)

## **🚀 Usage**

### **1. Data Preparation**
```python
# Generate masks from annotations
python src/03_mask_generation.py --json_path annotations/_annotations.coco.json

# Split dataset
python src/04_data_split.py --train_ratio 0.8
```

### **2. Training the Model**
```python
# Train UNet model
python src/05_training.py \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --augmentation True
```

### **3. Inference**
```python
# Segment new images
python src/06_inference.py \
    --model_path models/unet_segmentation_model.h5 \
    --image_path test_image.jpg \
    --output_path segmented_mask.png
```

### **4. Evaluation**
```python
# Evaluate model performance
python src/07_evaluation.py \
    --model_path models/unet_segmentation_model.h5 \
    --test_dir dataset/val/
```

## **📈 Results**

### **Performance Metrics**
| Metric | Training | Validation |
|--------|----------|------------|
| **Accuracy** | 96.2% | 94.7% |
| **IoU** | 89.5% | 87.3% |
| **Dice Coefficient** | 91.2% | 89.1% |
| **Precision** | 92.8% | 90.5% |
| **Recall** | 94.1% | 91.8% |

### **Visual Results**
![Segmentation Examples](docs/segmentation_examples.png)
*Left: Original image, Middle: Ground truth mask, Right: Predicted mask*

## **🔧 Customization**

### **Configuring Hyperparameters**
Edit `config.yaml`:
```yaml
model:
  input_size: [256, 256, 3]
  filters: [64, 128, 256, 512, 1024]
  
training:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0001
  
data:
  train_ratio: 0.8
  augmentation: true
  target_size: [256, 256]
```

### **Adding New Classes**
To extend for multi-class segmentation:
1. Update annotation format to include class labels
2. Modify mask generation to create multi-channel masks
3. Change loss function to categorical_crossentropy
4. Update final layer activation to softmax

## **📝 Key Implementation Details**

### **Mask Generation**
```python
def create_binary_mask(segmentations, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in segmentations:
        poly = np.array(seg).reshape((-1, 2))
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask * 255
```

### **Data Augmentation Pipeline**
```python
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
])
```

## **🤝 Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## **📄 License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **🙏 Acknowledgments**
- Open-source animal image repositories
- CVAT annotation tool team
- UNet original authors (Ronneberger et al.)
- TensorFlow and OpenCV communities

## **📧 Contact**
Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/animal-segmentation-unet](https://github.com/yourusername/animal-segmentation-unet)

## **🔗 References**
1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation
2. COCO Dataset: Common Objects in Context
3. CVAT: Computer Vision Annotation Tool
4. Albumentations: Fast image augmentation library

---

**⭐ If you find this project useful, please give it a star!**

---

## **Quick Start Guide**

### **For Quick Testing**
```bash
# Install
git clone https://github.com/yourusername/animal-segmentation-unet.git
pip install -r requirements.txt

# Download pretrained model
wget https://your-model-link/unet_segmentation_model.h5 -O models/

# Run inference
python src/06_inference.py --image_path your_image.jpg
```

### **For Developers**
```bash
# Run complete pipeline
./run_pipeline.sh

# Test with sample data
python tests/test_pipeline.py

# Generate documentation
pdoc --html src/ --output-dir docs/
```

## **Troubleshooting**

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size to 4 or 2 |
| Slow training | Use mixed precision training or reduce image size |
| Poor segmentation results | Increase augmentation, add more training data |
| Mask generation errors | Verify COCO JSON format and polygon coordinates |

---

*Last updated: December 2023*
