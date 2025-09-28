import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from typing import Tuple, Dict, List


class Trainer:
    """Training and evaluation functionality for neural networks."""

    def __init__(self, model, device, save_dir="./models_saved"):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, train_loader, criterion, optimizer, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print progress
            if batch_idx % 100 == 0:
                accuracy = 100.0 * correct / total
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {running_loss/(batch_idx+1):.3f} | '
                      f'Acc: {accuracy:.2f}% ({correct}/{total})')

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
              momentum=0.9, weight_decay=5e-4, save_best=True, save_checkpoint=False):
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: Weight decay for regularization
            save_best: Whether to save the best model
            save_checkpoint: Whether to save checkpoints during training
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")

        # Initialize optimizer and criterion
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
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # Save best model
            if save_best and val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model('best_model.pth')
                print(f'  New best accuracy: {val_acc:.2f}%')

            # Save checkpoint
            if save_checkpoint and (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, optimizer, scheduler, f'checkpoint_epoch_{epoch+1}.pth')

            print('-' * 60)

        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.2f} seconds')
        print(f'Best validation accuracy: {self.best_accuracy:.2f}%')

        return self.training_history

    def fine_tune(self, train_loader, val_loader, num_epochs=20, learning_rate=0.01):
        """
        Fine-tune the model (typically after pruning).

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of fine-tuning epochs
            learning_rate: Fine-tuning learning rate
        """
        print(f"Starting fine-tuning for {num_epochs} epochs...")

        # Use smaller learning rate for fine-tuning
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                             momentum=0.9, weight_decay=1e-4)

        # Simpler scheduler for fine-tuning
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, epoch)

            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)

            # Update learning rate
            scheduler.step()

            # Print epoch results
            print(f'Fine-tune Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

            # Update best accuracy
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc

        print(f'Fine-tuning completed. Best accuracy: {self.best_accuracy:.2f}%')
        return self.best_accuracy

    def test(self, test_loader):
        """Test the model."""
        print("Testing model...")
        test_loss, test_acc = self.validate(test_loader, nn.CrossEntropyLoss())
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        return test_acc

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

    def save_checkpoint(self, epoch, optimizer, scheduler, filename):
        """Save training checkpoint."""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filename, optimizer=None, scheduler=None):
        """Load training checkpoint."""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_accuracy = checkpoint['best_accuracy']
        self.training_history = checkpoint['training_history']

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from {filepath}")
        return checkpoint['epoch']


def get_lr_schedule(optimizer, schedule_type='multistep', **kwargs):
    """Get learning rate scheduler."""
    if schedule_type == 'multistep':
        milestones = kwargs.get('milestones', [50, 75])
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif schedule_type == 'cosine':
        T_max = kwargs.get('T_max', 100)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    elif schedule_type == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def count_parameters(model):
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calculate_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


