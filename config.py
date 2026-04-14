import torch
import os

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Learning rate scheduler
LR_STEP_SIZE = 5    # Decay LR every N epochs
LR_GAMMA = 0.5      # Multiplicative factor for LR decay

# Model parameters
NUM_CLASSES = 10        # Digits 0-9 (use 47 for EMNIST balanced)
HIDDEN_UNITS = 128
DROPOUT_RATE = 0.5

# DataLoader
NUM_WORKERS = 2
RANDOM_SEED = 42

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
PLOT_PATH = os.path.join(MODEL_DIR, "training_curves.png")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset: 'MNIST' or 'EMNIST'
DATASET = "MNIST"
