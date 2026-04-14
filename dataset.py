import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config


def get_transforms():
    """Return normalization transforms for MNIST/EMNIST images."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])


def get_dataloaders(dataset_name=config.DATASET, val_split=0.1):
    """
    Load MNIST or EMNIST dataset and return train, validation, and test dataloaders.

    Args:
        dataset_name (str): 'MNIST' or 'EMNIST'
        val_split (float): Fraction of training data used for validation.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    transform = get_transforms()

    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            root=config.DATA_DIR, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=config.DATA_DIR, train=False, download=True, transform=transform
        )
    elif dataset_name == "EMNIST":
        train_dataset = datasets.EMNIST(
            root=config.DATA_DIR,
            split="balanced",
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.EMNIST(
            root=config.DATA_DIR,
            split="balanced",
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'MNIST' or 'EMNIST'.")

    # Split training data into train and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader
