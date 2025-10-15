"""
Teacher Model Training Module

Trains teacher models (ResNet-18 and ViT-Tiny) on CIFAR-10.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from typing import Tuple


class TeacherTrainer:
    """Trainer for teacher models."""

    def __init__(self, model, device, save_dir="./models_saved", model_name="teacher"):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, train_loader, criterion, optimizer, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                accuracy = 100.0 * correct / total
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {running_loss/(batch_idx+1):.3f} | Acc: {accuracy:.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader, criterion):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        return val_loss, val_acc

    def train(self, train_loader, val_loader, num_epochs=100, learning_rate=0.1,
              momentum=0.9, weight_decay=5e-4):
        """
        Train the teacher model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: Weight decay for regularization

        Returns:
            Best validation accuracy achieved
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                             momentum=momentum, weight_decay=weight_decay)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

        start_time = time.time()

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, epoch)

            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)

            # Update learning rate
            scheduler.step()

            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            # Print epoch results
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model(f'{self.model_name}.pth')
                print(f'  *** New best accuracy: {val_acc:.2f}% ***')

            print('-' * 60)

        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/60:.2f} minutes')
        print(f'Best validation accuracy: {self.best_accuracy:.2f}%')

        return self.best_accuracy

    def save_model(self, filename):
        """Save model state dict."""
        filepath = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filename):
        """Load model state dict."""
        filepath = os.path.join(self.save_dir, filename)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")


def train_teacher_model(model, train_loader, test_loader, device,
                       model_name="teacher", save_dir="./models_saved",
                       num_epochs=100, learning_rate=0.1):
    """
    Convenience function to train a teacher model.

    Args:
        model: Teacher model to train
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to train on
        model_name: Name for saving the model
        save_dir: Directory to save models
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate

    Returns:
        Tuple of (trained_model, best_accuracy)
    """
    trainer = TeacherTrainer(model, device, save_dir, model_name)
    best_acc = trainer.train(train_loader, test_loader,
                            num_epochs=num_epochs,
                            learning_rate=learning_rate)

    return model, best_acc
