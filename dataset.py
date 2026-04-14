import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_mnist_dataset(data_dir, train=True):
    """Load the MNIST dataset."""
    dataset = datasets.MNIST(root=data_dir, train=train, download=True)
    return dataset

def normalize_images(dataset):
    """Normalize dataset images to [0, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset.transform = transform
    return dataset

def create_dataloaders(dataset, batch_size=64, val_split=0.2):
    """Create training, validation, and testing dataloaders."""
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets.MNIST(root=data_dir, train=False, download=True), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    data_dir = './data'
    mnist_dataset = load_mnist_dataset(data_dir)
    mnist_dataset = normalize_images(mnist_dataset)
    train_loader, val_loader, test_loader = create_dataloaders(mnist_dataset)