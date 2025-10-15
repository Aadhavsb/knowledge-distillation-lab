"""
Knowledge Distillation Loss Functions

Implements various KD loss functions including:
- Soft-target KD with temperature scaling
- Combined loss (hard target + soft target)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss combining hard and soft targets.

    Based on Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)

    Loss = alpha * CE(y, student_logits) + (1 - alpha) * T^2 * KL(teacher_soft || student_soft)

    where:
    - y: true labels (hard targets)
    - student_logits: logits from student model
    - teacher_soft: softmax(teacher_logits / T)
    - student_soft: softmax(student_logits / T)
    - T: temperature (controls softness of probability distribution)
    - alpha: weight for hard target loss
    """

    def __init__(self, alpha=0.5, temperature=4.0):
        """
        Args:
            alpha (float): Weight for hard target loss. Default: 0.5
            temperature (float): Temperature for softening distributions. Default: 4.0
        """
        super(KDLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        """
        Compute KD loss.

        Args:
            student_logits: Logits from student model (B, num_classes)
            teacher_logits: Logits from teacher model (B, num_classes)
            labels: Ground truth labels (B,)

        Returns:
            Total KD loss
        """
        # Hard target loss (cross-entropy with true labels)
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft target loss (KL divergence between teacher and student distributions)
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return total_loss, hard_loss, soft_loss


def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):
    """
    Functional interface for KD loss.

    Args:
        student_logits: Logits from student model (B, num_classes)
        teacher_logits: Logits from teacher model (B, num_classes)
        labels: Ground truth labels (B,)
        alpha (float): Weight for hard target loss
        temperature (float): Temperature for softening distributions

    Returns:
        tuple: (total_loss, hard_loss, soft_loss)
    """
    kd_loss_fn = KDLoss(alpha=alpha, temperature=temperature)
    return kd_loss_fn(student_logits, teacher_logits, labels)


def kl_divergence_loss(student_logits, teacher_logits, temperature=4.0):
    """
    Pure KL divergence loss between teacher and student (no hard targets).

    Args:
        student_logits: Logits from student model (B, num_classes)
        teacher_logits: Logits from teacher model (B, num_classes)
        temperature (float): Temperature for softening distributions

    Returns:
        KL divergence loss
    """
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

    kl_div = nn.KLDivLoss(reduction='batchmean')
    loss = kl_div(student_soft, teacher_soft) * (temperature ** 2)

    return loss


def soft_target_cross_entropy(student_logits, teacher_logits, temperature=4.0):
    """
    Alternative soft target loss using cross-entropy.

    Args:
        student_logits: Logits from student model (B, num_classes)
        teacher_logits: Logits from teacher model (B, num_classes)
        temperature (float): Temperature for softening distributions

    Returns:
        Soft cross-entropy loss
    """
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

    loss = -torch.mean(torch.sum(teacher_soft * student_soft, dim=1))

    return loss * (temperature ** 2)
