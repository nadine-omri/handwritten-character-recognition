"""Fonctions utilitaires"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

def set_seed(seed):
    """Fixer la graine pour la reproductibilité"""
    torch.manual_seed(seed)
    np.random.seed(seed)

def plot_images(images, labels, num_images=9):
    """Afficher des images d'entraînement"""
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        axes[i].imshow(images[i].numpy().squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_device():
    """Déterminer le device (GPU ou CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(outputs, labels):
    """Calculer la précision"""
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total