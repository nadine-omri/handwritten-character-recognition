# 🖊️ Handwritten Character Recognition

A deep learning project for recognizing handwritten digits and characters using Convolutional Neural Networks (CNN).

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/nadine-omri/handwritten-character-recognition.git
cd handwritten-character-recognition
pip install -r requirements.txt
```

### 2. Train Model
```bash
python train.py
```

### 3. Make Predictions
```bash
python inference.py --image path/to/digit.png
```

## 📁 Project Structure

- `config.py` - Configuration & hyperparameters
- `dataset.py` - MNIST data loading & preprocessing
- `model.py` - CNN architectures (SimpleCNN, DeepCNN)
- `train.py` - Training pipeline
- `inference.py` - Prediction on new images
- `requirements.txt` - Dependencies

## ✅ Features

- MNIST dataset support (60,000 training samples)
- Extendable to EMNIST (47 classes - letters + digits)
- SimpleCNN (2 layers) for quick training
- DeepCNN (3 layers) for better accuracy
- Model persistence & batch inference
- Training visualization

## 📊 Expected Results

- SimpleCNN: 98% accuracy in ~90 seconds
- DeepCNN: 99%+ accuracy in ~120 seconds

## 🔄 Roadmap

- Phase 1: ✅ MVP Complete (MNIST + SimpleCNN)
- Phase 2: Add more CNN layers & ResNet blocks
- Phase 3: EMNIST support (full alphabet)
- Phase 4: CRNN for word recognition
- Phase 5: Production deployment

## 🛠️ Dependencies

- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Pillow

## 👤 Author

Nadine Omri - [GitHub](https://github.com/nadine-omri)

---

**Happy Learning! 🚀**