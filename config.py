"""Configuration du projet"""

# Chemins
DATA_DIR = "./data/mnist"
MODEL_DIR = "./models"

# Paramètres du modèle
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10  # Chiffres 0-9
HIDDEN_SIZE = 128

# Paramètres d'entraînement
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DEVICE = "cuda"  # ou "cpu" si pas de GPU

# Paramètres de données
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42