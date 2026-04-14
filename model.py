import torch
import torch.nn as nn


class CharRecognitionCNN(nn.Module):
    """
    Convolutional Neural Network for handwritten character recognition.

    Architecture:
        - Conv Block 1: 32 filters, 3x3 kernel + BatchNorm + ReLU + MaxPool
        - Conv Block 2: 64 filters, 3x3 kernel + BatchNorm + ReLU + MaxPool
        - Fully Connected: 64*7*7 -> hidden_units -> num_classes
        - Dropout for regularization
    """

    def __init__(self, num_classes=10, hidden_units=128, dropout_rate=0.5):
        """
        Args:
            num_classes (int): Number of output classes (10 for MNIST, 47 for EMNIST balanced).
            hidden_units (int): Number of units in the hidden fully-connected layer.
            dropout_rate (float): Dropout probability.
        """
        super(CharRecognitionCNN, self).__init__()

        # Convolutional block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
        )

        # Convolutional block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=10, hidden_units=128, dropout_rate=0.5, device=None):
    """
    Build and return a CharRecognitionCNN model.

    Args:
        num_classes (int): Number of output classes.
        hidden_units (int): Hidden layer size.
        dropout_rate (float): Dropout probability.
        device (torch.device): Target device.

    Returns:
        CharRecognitionCNN: Model moved to the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharRecognitionCNN(num_classes=num_classes, hidden_units=hidden_units, dropout_rate=dropout_rate)
    return model.to(device)
