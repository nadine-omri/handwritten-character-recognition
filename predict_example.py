import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.predict import CharacterPredictor
import config

# Initialiser le prédicteur
predictor = CharacterPredictor(
    model_path=f"{config.MODEL_DIR}/best_model_advanced.pt",
    model_type='advanced'
)

# Prédire sur une image (remplace le chemin)
# ✅ Correct
pred_class, confidence, probs = predictor.predict('C:/Users/nadin/Downloads/OIP (1).webp')
print(f"Prédiction: {pred_class}")
print(f"Confiance: {confidence:.4f}")