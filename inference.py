import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import config
from model import build_model

# Class labels for MNIST digits; extend for EMNIST if needed
MNIST_CLASSES = [str(i) for i in range(10)]


def load_model(model_path=config.BEST_MODEL_PATH, num_classes=config.NUM_CLASSES, device=None):
    """
    Load a trained model from a checkpoint file.

    Args:
        model_path (str): Path to the saved model weights (.pth file).
        num_classes (int): Number of output classes.
        device (torch.device): Target device.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode.
    """
    if device is None:
        device = config.DEVICE
    model = build_model(
        num_classes=num_classes,
        hidden_units=config.HIDDEN_UNITS,
        dropout_rate=config.DROPOUT_RATE,
        device=device,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def preprocess_image(image):
    """
    Preprocess an input image for inference.

    Args:
        image: PIL.Image, numpy array (H x W) or (H x W x C), or file path (str).

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 1, 28, 28).
    """
    if isinstance(image, str):
        image = Image.open(image).convert("L")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("L")
    elif isinstance(image, Image.Image):
        image = image.convert("L")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict(image, model=None, class_labels=None, device=None):
    """
    Predict the class of a handwritten character image.

    Args:
        image: PIL.Image, numpy array, or file path (str).
        model (torch.nn.Module): Trained model. Loaded automatically if None.
        class_labels (list): List of class label strings.
        device (torch.device): Target device.

    Returns:
        dict: {
            'predicted_class': str,
            'predicted_index': int,
            'confidence': float,
            'probabilities': list of (label, probability) tuples sorted by probability
        }
    """
    if device is None:
        device = config.DEVICE
    if model is None:
        model = load_model(device=device)
    if class_labels is None:
        class_labels = MNIST_CLASSES

    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_labels[predicted_index]
    confidence = float(probabilities[predicted_index])

    sorted_probs = sorted(
        zip(class_labels, probabilities.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "confidence": confidence,
        "probabilities": sorted_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict handwritten character from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--model", type=str, default=config.BEST_MODEL_PATH, help="Path to model checkpoint.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display.")
    args = parser.parse_args()

    model = load_model(model_path=args.model)
    result = predict(args.image, model=model)

    print(f"\nPredicted class : {result['predicted_class']}")
    print(f"Confidence      : {result['confidence'] * 100:.2f}%")
    print(f"\nTop-{args.top_k} predictions:")
    for label, prob in result["probabilities"][: args.top_k]:
        bar = "#" * int(prob * 30)
        print(f"  {label}: {prob * 100:6.2f}%  {bar}")


if __name__ == "__main__":
    main()
