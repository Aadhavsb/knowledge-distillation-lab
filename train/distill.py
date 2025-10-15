"""
Knowledge Distillation Training Module

Implements student training with knowledge distillation from teacher models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from utils.kd_losses import KDLoss


class DistillationTrainer:
    """Trainer for student models using knowledge distillation."""

    def __init__(self, student_model, teacher_model, device,
                 save_dir="./models_saved", model_name="student"):
        self.student = student_model
        self.teacher = teacher_model
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'hard_loss': [],
            'soft_loss': []
        }

        # Set teacher to eval mode
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, train_loader, kd_criterion, optimizer, epoch):
        """Train for one epoch with distillation."""
        self.student.train()
        self.teacher.eval()

        running_loss = 0.0
        running_hard_loss = 0.0
        running_soft_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)

            # Get student predictions
            student_logits = self.student(inputs)

            # Calculate KD loss
            loss, hard_loss, soft_loss = kd_criterion(student_logits, teacher_logits, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            running_hard_loss += hard_loss.item()
            running_soft_loss += soft_loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                accuracy = 100.0 * correct / total
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {running_loss/(batch_idx+1):.3f} | '
                      f'Hard: {running_hard_loss/(batch_idx+1):.3f} | '
                      f'Soft: {running_soft_loss/(batch_idx+1):.3f} | '
                      f'Acc: {accuracy:.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_hard_loss = running_hard_loss / len(train_loader)
        epoch_soft_loss = running_soft_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc, epoch_hard_loss, epoch_soft_loss

    def validate(self, val_loader):
        """Validate the student model."""
        self.student.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.student(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        return val_loss, val_acc

    def distill(self, train_loader, val_loader, num_epochs=100, learning_rate=0.1,
                momentum=0.9, weight_decay=5e-4, alpha=0.5, temperature=4.0):
        """
        Train student model with knowledge distillation.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: Weight decay for regularization
            alpha: Weight for hard target loss (0 to 1)
            temperature: Temperature for soft targets

        Returns:
            Best validation accuracy achieved
        """
        print(f"\n{'='*60}")
        print(f"Distilling knowledge to {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Student parameters: {sum(p.numel() for p in self.student.parameters()):,}")
        print(f"Teacher parameters: {sum(p.numel() for p in self.teacher.parameters()):,}")
        print(f"Alpha: {alpha}, Temperature: {temperature}")

        # KD loss
        kd_criterion = KDLoss(alpha=alpha, temperature=temperature)

        # Optimizer
        optimizer = optim.SGD(self.student.parameters(), lr=learning_rate,
                             momentum=momentum, weight_decay=weight_decay)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

        start_time = time.time()

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc, hard_loss, soft_loss = self.train_epoch(
                train_loader, kd_criterion, optimizer, epoch
            )

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate
            scheduler.step()

            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['hard_loss'].append(hard_loss)
            self.training_history['soft_loss'].append(soft_loss)

            # Print epoch results
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f} (Hard: {hard_loss:.4f}, Soft: {soft_loss:.4f})')
            print(f'  Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model(f'{self.model_name}.pth')
                print(f'  *** New best accuracy: {val_acc:.2f}% ***')

            print('-' * 60)

        total_time = time.time() - start_time
        print(f'\nDistillation completed in {total_time/60:.2f} minutes')
        print(f'Best validation accuracy: {self.best_accuracy:.2f}%')

        return self.best_accuracy

    def save_model(self, filename):
        """Save student model state dict."""
        filepath = os.path.join(self.save_dir, filename)
        torch.save(self.student.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filename):
        """Load student model state dict."""
        filepath = os.path.join(self.save_dir, filename)
        self.student.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")


def train_student_with_kd(student_model, teacher_model, train_loader, test_loader,
                          device, model_name="student", save_dir="./models_saved",
                          num_epochs=100, learning_rate=0.1, alpha=0.5, temperature=4.0):
    """
    Convenience function to train a student model with knowledge distillation.

    Args:
        student_model: Student model to train
        teacher_model: Trained teacher model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to train on
        model_name: Name for saving the model
        save_dir: Directory to save models
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        alpha: Weight for hard target loss
        temperature: Temperature for soft targets

    Returns:
        Tuple of (trained_student, best_accuracy)
    """
    trainer = DistillationTrainer(student_model, teacher_model, device, save_dir, model_name)
    best_acc = trainer.distill(train_loader, test_loader,
                               num_epochs=num_epochs,
                               learning_rate=learning_rate,
                               alpha=alpha,
                               temperature=temperature)

    return student_model, best_acc


def train_student_without_kd(student_model, train_loader, test_loader, device,
                             model_name="student", save_dir="./models_saved",
                             num_epochs=100, learning_rate=0.1):
    """
    Train a student model WITHOUT knowledge distillation (baseline).

    Args:
        student_model: Student model to train
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to train on
        model_name: Name for saving the model
        save_dir: Directory to save models
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate

    Returns:
        Tuple of (trained_student, best_accuracy)
    """
    from train.train_teacher import TeacherTrainer

    trainer = TeacherTrainer(student_model, device, save_dir, model_name)
    best_acc = trainer.train(train_loader, test_loader,
                            num_epochs=num_epochs,
                            learning_rate=learning_rate)

    return student_model, best_acc
