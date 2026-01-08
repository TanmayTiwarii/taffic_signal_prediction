# 🚦 Traffic Sign Detection using OpenCV & PyTorch

A deep learning–based traffic sign classification system built using **PyTorch** and **OpenCV**.  
The model is trained on a multi-class traffic sign dataset and achieves **~85% accuracy on the test set** after proper dataset balancing and evaluation.

---

## 📌 Overview
This project focuses on building a **robust traffic sign recognition system** while addressing real-world challenges such as:

- Class imbalance
- Train–test label mismatch
- Proper multi-class evaluation

The system uses a **ResNet-18** model and follows clean machine learning practices.

---

## 🧠 Tech Stack
- Python  
- PyTorch  
- Torchvision  
- OpenCV  
- NumPy  
- Matplotlib  
- Seaborn  

---

## 📂 Project Structure
```
archive/
├── train.py                  # Model training
├── test.py                   # Model testing
├── confusion_matrix.py       # Confusion matrix generation
├── rebuild_test_sorted.py    # Fixes test dataset structure
├── traffic_sign_model.pth    # Trained model
├── dataset/
│   └── traffic_Data/
│       ├── DATA/             # Raw training data
│       ├── TEST/             # Raw test images
│       └── TEST_SORTED/      # Processed test data
└── dataset_balanced/
    └── traffic_Data/
        └── DATA/             # Balanced training dataset
```

---

## 📊 Dataset
- **Training Dataset**: Balanced dataset with ~100 images per class  
- **Classes Used**: ~47 (extremely small classes removed)  
- **Test Dataset**: Rebuilt by extracting class labels from image filenames  

Class imbalance was handled using **targeted data augmentation**.

---

## 🚀 Model Details
- **Architecture**: ResNet-18 (pretrained)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Input Size**: 64 × 64 RGB
- **Train / Validation Split**: 80% / 20%

---

## 📈 Results
- **Test Accuracy**: ~85%
- **Confusion Matrix**: Strong diagonal dominance
- Most misclassifications occur between visually similar traffic signs

---

## 🧪 Evaluation
- Test dataset was cleaned to include only classes seen during training
- Confusion matrix used for detailed per-class performance analysis

---
## YOLOv8 Detection
- Model: YOLOv8n
- Dataset: Converted from classification to detection (weak supervision)
- Real-time inference supported via webcam


## 🛠️ How to Run

### 1️⃣ Install dependencies
```bash
pip install torch torchvision numpy pillow scikit-learn matplotlib seaborn opencv-python
```

### 2️⃣ Train the model
```bash
python train.py
```

### 3️⃣ Rebuild test dataset (if required)
```bash
python rebuild_test_sorted.py
```

### 4️⃣ Evaluate and generate confusion matrix
```bash
python confusion_matrix.py
```

---

## 🧠 Key Learnings
- Handling severe class imbalance in real-world datasets  
- Fixing class-index mismatches between training and testing  
- Proper evaluation of multi-class deep learning models  
- Practical integration of PyTorch with OpenCV  

---

## 📌 Future Work
- Real-time traffic sign recognition using webcam (OpenCV)
- Faster inference using MobileNet
- Bounding-box based detection using YOLO
- Deployment as a mobile or web application

---

## 👤 Author
**Tanmay Tiwari**  
B.Tech CSE | Computer Vision & Deep Learning
