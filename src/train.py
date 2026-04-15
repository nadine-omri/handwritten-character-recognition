"""Script d'entraînement"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import CNN, AdvancedCNN
from src.data_loader import load_mnist_data
from src.utils import set_seed, accuracy, get_device
import os

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîner une époque"""
    model.train()
    total_loss = 0
    total_acc = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    
    return avg_loss, avg_acc

def evaluate(model, test_loader, criterion, device):
    """Évaluer le modèle"""
    model.eval()
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)
    
    avg_loss = total_loss / len(test_loader)
    avg_acc = total_acc / len(test_loader)
    
    return avg_loss, avg_acc

def train(model_type='basic', num_epochs=config.NUM_EPOCHS):
    """Fonction principale d'entraînement"""
    
    # Initialisation
    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"Using device: {device}")
    
    # Charger les données
    print("Loading data...")
    train_loader, test_loader, _, _ = load_mnist_data()
    
    # Créer le modèle
    if model_type == 'basic':
        model = CNN().to(device)
    elif model_type == 'advanced':
        model = AdvancedCNN().to(device)
    else:
        raise ValueError("model_type doit être 'basic' ou 'advanced'")
    
    print(f"Model: {model_type}")
    print(f"Paramètres totaux: {sum(p.numel() for p in model.parameters())}")
    
    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Entraînement
    best_val_acc = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Époque [{epoch+1}/{num_epochs}]")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{config.MODEL_DIR}/best_model_{model_type}.pt")
            print(f"✓ Meilleur modèle sauvegardé (acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping après {epoch+1} époque(s)")
                break
    
    print(f"\n{'='*50}")
    print(f"Entraînement terminé!")
    print(f"Meilleure précision: {best_val_acc:.4f}")

if __name__ == "__main__":
    train(model_type='advanced', num_epochs=config.NUM_EPOCHS)