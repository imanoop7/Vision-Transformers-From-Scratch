"""
Training script for the Vision Transformer model.

This script handles:
1. Data loading and preprocessing
2. Model initialization
3. Training loop with metrics tracking
4. Optimization using AdamW
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vit import VisionTransformer

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The Vision Transformer model
        dataloader (DataLoader): DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        device: Device to train on (cuda/cpu)
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(dataloader), 100. * correct / total

def main():
    """
    Main training function.
    
    Sets up the training environment, initializes the model,
    and runs the training loop.
    """
    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 3e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize(224),        # Resize images to ViT input size
        transforms.CenterCrop(224),    # Crop to remove any resize artifacts
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    trainset = datasets.CIFAR10(root='./data', train=True,
                               download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                           shuffle=True, num_workers=2)
    
    # Initialize model
    print(f"Initializing Vision Transformer on {device}...")
    model = VisionTransformer(
        image_size=224,      # Match the transform size
        patch_size=16,       # 16x16 patches
        in_channels=3,       # RGB images
        num_classes=10,      # CIFAR-10 classes
        embed_dim=768,       # Embedding dimension
        depth=12,            # Number of transformer layers
        num_heads=12,        # Number of attention heads
        mlp_ratio=4.0,       # MLP hidden dim ratio
        dropout=0.1          # Dropout rate
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch+1}/{num_epochs}')
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device
        )
        print(f'Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.2f}%')

if __name__ == '__main__':
    main()
