import torch
import os

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
MOMENTUM = 0.9

# Model parameters
INPUT_CHANNELS = 1
NUM_CLASSES = 10  # 0-9 for MNIST, 47 for EMNIST
HIDDEN_UNITS = 128
DROPOUT_RATE = 0.5

# Data paths
DATA_DIR = './data'
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Model save path
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'checkpoint.pth')

# Dataset configuration
DATASET_TYPE = 'MNIST'  # 'MNIST' or 'EMNIST'
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Logging
PRINT_EVERY = 100
VISUALIZE_TRAINING = True
