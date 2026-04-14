import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

import config
from dataset import get_dataloaders
from model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run a single training epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataloader. Returns average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training/validation loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Accuracy")
    ax2.plot(epochs, val_accs, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def train():
    """Main training function."""
    device = config.DEVICE
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders()
    print(
        f"Dataset: {config.DATASET} | "
        f"Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    # Model
    model = build_model(
        num_classes=config.NUM_CLASSES,
        hidden_units=config.HIDDEN_UNITS,
        dropout_rate=config.DROPOUT_RATE,
        device=device,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch:2d}/{config.NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%"
        )

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  -> Best model saved (Val Acc: {best_val_acc:.2f}%)")

    # Plot curves
    plot_curves(train_losses, val_losses, train_accs, val_accs, config.PLOT_PATH)

    # Final test evaluation using the best model
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    train()
