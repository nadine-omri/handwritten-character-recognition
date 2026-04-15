"""Chargement et préparation des données"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config

def get_transforms():
    """Définir les transformations des images"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def load_mnist_data(batch_size=config.BATCH_SIZE):
    """Charger le dataset MNIST"""
    
    transform = get_transforms()
    
    # Télécharger et charger les données
    train_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader, train_dataset, test_dataset

def load_emnist_data(batch_size=config.BATCH_SIZE):
    """Charger le dataset EMNIST (caractères)"""
    
    transform = get_transforms()
    
    train_dataset = datasets.EMNIST(
        root=config.DATA_DIR,
        split='balanced',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.EMNIST(
        root=config.DATA_DIR,
        split='balanced',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader, train_dataset, test_dataset