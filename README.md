# Handwritten Character Recognition

Handwritten Character Recognition using Convolutional Neural Networks (CNN) trained on MNIST (digits 0–9), with a clean architecture that is easily extensible to EMNIST (letters + digits).

---

## 📁 Project Structure

```
handwritten-character-recognition/
├── config.py          # Hyperparameters and paths
├── dataset.py         # MNIST / EMNIST data loading & preprocessing
├── model.py           # CNN architecture
├── train.py           # Training loop with validation and checkpointing
├── inference.py       # Prediction on new images
├── requirements.txt   # Python dependencies
└── README.md
```

---

## 🧠 Model Architecture

```
Input (1×28×28)
│
├── Conv Block 1: Conv2d(1→32, 3×3) + BatchNorm + ReLU + MaxPool  → (32×14×14)
├── Conv Block 2: Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool → (64×7×7)
│
├── Flatten → 3136
├── Linear(3136 → 128) + ReLU + Dropout(0.5)
└── Linear(128 → 10)   # output logits
```

- **Parameters**: ~430 K — trains in minutes on CPU
- **Regularization**: BatchNorm + Dropout
- **Extensible**: change `NUM_CLASSES` in `config.py` for EMNIST (47 classes)

---

## 🚀 Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/nadine-omri/handwritten-character-recognition.git
cd handwritten-character-recognition
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

MNIST is downloaded automatically to `data/`. The best model checkpoint is saved to
`models/best_model.pth` and training curves are saved to `models/training_curves.png`.

### 3. Predict on a new image

```bash
python inference.py --image path/to/digit.png
```

Example output:

```
Predicted class : 7
Confidence      : 99.23%

Top-3 predictions:
  7:  99.23%  ##############################
  1:   0.51%
  9:   0.18%
```

---

## ⚙️ Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 64 | Mini-batch size |
| `LEARNING_RATE` | 0.001 | Adam optimizer LR |
| `NUM_EPOCHS` | 10 | Training epochs |
| `NUM_CLASSES` | 10 | Output classes (10=MNIST, 47=EMNIST) |
| `HIDDEN_UNITS` | 128 | FC hidden layer size |
| `DROPOUT_RATE` | 0.5 | Dropout probability |
| `DATASET` | `"MNIST"` | `"MNIST"` or `"EMNIST"` |

---

## 🔬 Inference API

```python
from inference import load_model, predict

model = load_model("models/best_model.pth")
result = predict("digit.png", model=model)

print(result["predicted_class"])   # e.g. "7"
print(result["confidence"])        # e.g. 0.9923
print(result["probabilities"])     # sorted list of (label, prob)
```

`predict()` accepts a file path (`str`), a PIL `Image`, or a NumPy array.

---

## 📈 Next Steps

1. **EMNIST support** — set `DATASET = "EMNIST"` and `NUM_CLASSES = 47` in `config.py`
2. **Data augmentation** — random rotation, affine transforms for better generalization
3. **Deeper CNN** — add a third conv block for higher accuracy
4. **Sequence modeling** — CRNN (CNN + LSTM/CTC) for full word/sentence recognition
5. **Web demo** — Flask or Gradio interface for live digit drawing
