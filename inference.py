import torch
import torch.nn.functional as F

class Predictor:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, image):
        # Preprocess the image (this depends on your specific model requirements)
        image = self.preprocess(image)
        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)  # Get probabilities for each class
            confidence_scores, predicted_classes = torch.max(probabilities, 1)
            return predicted_classes, confidence_scores, probabilities

    def preprocess(self, image):
        # Implement your image preprocessing here
        # For example: resizing, normalization, etc.
        return image