import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import config
from model import get_model
from torchvision import transforms
import argparse

class Predictor:
    def __init__(self, model_path, model_type='simple'):
        self.device = config.DEVICE
        self.model = get_model(model_type, config.NUM_CLASSES).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def predict(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
        
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        return prediction.item(), confidence.item(), probabilities.cpu().numpy()[0]

def predict_mnist_sample():
    from torchvision import datasets
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root=config.DATA_DIR, train=False, download=True, transform=transform)
    image, label = test_dataset[0]
    
    predictor = Predictor(config.BEST_MODEL_PATH, model_type='simple')
    
    image = image.unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        output = predictor.model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    print(f"True Label: {label}")
    print(f"Predicted Label: {prediction.item()}")
    print(f"Confidence: {confidence.item():.2%}")
    print(f"Probabilities: {probabilities.cpu().numpy()[0]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict handwritten digit')
    parser.add_argument('--image', type=str, default=None, help='Path to image file')
    parser.add_argument('--model', type=str, default='simple', help='Model type: simple or deep')
    
    args = parser.parse_args()
    
    if args.image:
        predictor = Predictor(config.BEST_MODEL_PATH, model_type=args.model)
        prediction, confidence, probabilities = predictor.predict(args.image)
        print(f"Predicted digit: {prediction}")
        print(f"Confidence: {confidence:.2%}")
    else:
        print("No trained model found. Please train first with: python train.py")
        print("\nUsage: python inference.py --image path/to/image.png --model simple")