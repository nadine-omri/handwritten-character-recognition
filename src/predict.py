"""Script de prédiction"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import config
from src.model import CNN, AdvancedCNN
from src.utils import get_device

class CharacterPredictor:
    """Classe pour faire des prédictions"""
    
    def __init__(self, model_path, model_type='advanced'):
        self.device = get_device()
        
        # Charger le modèle
        if model_type == 'basic':
            self.model = CNN().to(self.device)
        elif model_type == 'advanced':
            self.model = AdvancedCNN().to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Définir les transformations
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def predict(self, image_path):
        """Prédire le caractère d'une image"""
        # Charger l'image
        image = Image.open(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, probabilities.cpu().numpy()[0]
    
    def predict_batch(self, image_paths):
        """Prédire plusieurs images"""
        results = []
        for path in image_paths:
            pred_class, confidence, probs = self.predict(path)
            results.append({
                'path': path,
                'prediction': pred_class,
                'confidence': confidence,
                'probabilities': probs
            })
        return results

if __name__ == "__main__":
    predictor = CharacterPredictor(
        model_path=f"{config.MODEL_DIR}/best_model_advanced.pt",
        model_type='advanced'
    )
    print("Predictor initialized successfully!")