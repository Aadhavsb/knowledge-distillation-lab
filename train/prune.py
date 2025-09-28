import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict


class CustomPruner:
    """Custom pruning implementation for neural networks."""

    def __init__(self, model):
        self.model = model
        self.original_weights = {}
        self.masks = {}

    def save_original_weights(self):
        """Save original weights before pruning."""
        self.original_weights = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.original_weights[name] = param.data.clone()

    def calculate_sparsity(self):
        """Calculate overall sparsity of the model."""
        total_params = 0
        zero_params = 0

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()

        sparsity = zero_params / total_params if total_params > 0 else 0
        return sparsity

    def magnitude_based_unstructured_pruning(self, pruning_ratio: float):
        """
        Perform magnitude-based unstructured pruning.

        Args:
            pruning_ratio (float): Fraction of weights to prune (0-1)
        """
        print(f"Performing unstructured pruning with ratio: {pruning_ratio:.2f}")

        # Collect all weights for global magnitude ranking
        all_weights = []
        weight_info = []

        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weights_flat = param.data.abs().flatten()
                all_weights.append(weights_flat)
                weight_info.append((name, param.shape, len(weights_flat)))

        # Concatenate all weights
        all_weights_tensor = torch.cat(all_weights)

        # Find threshold for pruning
        k = int(len(all_weights_tensor) * pruning_ratio)
        if k > 0:
            threshold = torch.topk(all_weights_tensor, k, largest=False)[0][-1]
        else:
            threshold = 0

        # Apply pruning
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                mask = (param.data.abs() > threshold).float()
                param.data.mul_(mask)
                self.masks[name] = mask

        sparsity = self.calculate_sparsity()
        print(f"Achieved sparsity: {sparsity:.4f}")

    def iterative_magnitude_pruning(self, target_sparsity: float, num_iterations: int = 10):
        """
        Perform iterative magnitude-based pruning.

        Args:
            target_sparsity (float): Target sparsity level (0-1)
            num_iterations (int): Number of pruning iterations
        """
        print(f"Performing iterative pruning to {target_sparsity:.2f} sparsity in {num_iterations} iterations")

        current_sparsity = 0
        for iteration in range(num_iterations):
            # Calculate pruning ratio for this iteration
            remaining_weights = 1 - current_sparsity
            iteration_target = target_sparsity ** ((iteration + 1) / num_iterations)
            iteration_ratio = (iteration_target - current_sparsity) / remaining_weights

            if iteration_ratio <= 0:
                break

            self.magnitude_based_unstructured_pruning(iteration_ratio)
            current_sparsity = self.calculate_sparsity()

            print(f"Iteration {iteration + 1}: Current sparsity = {current_sparsity:.4f}")

            if current_sparsity >= target_sparsity:
                break

    def structured_channel_pruning(self, pruning_ratio: float):
        """
        Perform structured channel-wise pruning.

        Args:
            pruning_ratio (float): Fraction of channels to prune (0-1)
        """
        print(f"Performing structured channel pruning with ratio: {pruning_ratio:.2f}")

        pruned_channels_info = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
                weight = module.weight.data

                if isinstance(module, nn.Conv2d):
                    # For Conv2d: weight shape is (out_channels, in_channels, kernel_h, kernel_w)
                    num_channels = weight.shape[0]
                    num_to_prune = int(num_channels * pruning_ratio)

                    if num_to_prune > 0:
                        # Calculate channel importance (L2 norm of each output channel)
                        channel_importance = torch.norm(weight.view(num_channels, -1), p=2, dim=1)
                        _, indices_to_prune = torch.topk(channel_importance, num_to_prune, largest=False)

                        # Zero out the channels
                        weight[indices_to_prune] = 0

                        # If there's a bias, zero it out too
                        if module.bias is not None:
                            module.bias.data[indices_to_prune] = 0

                        pruned_channels_info[name] = {
                            'total_channels': num_channels,
                            'pruned_channels': num_to_prune,
                            'pruned_indices': indices_to_prune.tolist()
                        }

                elif isinstance(module, nn.Linear):
                    # For Linear: weight shape is (out_features, in_features)
                    num_features = weight.shape[0]
                    num_to_prune = int(num_features * pruning_ratio)

                    if num_to_prune > 0:
                        # Calculate feature importance
                        feature_importance = torch.norm(weight, p=2, dim=1)
                        _, indices_to_prune = torch.topk(feature_importance, num_to_prune, largest=False)

                        # Zero out the features
                        weight[indices_to_prune] = 0

                        # If there's a bias, zero it out too
                        if module.bias is not None:
                            module.bias.data[indices_to_prune] = 0

                        pruned_channels_info[name] = {
                            'total_features': num_features,
                            'pruned_features': num_to_prune,
                            'pruned_indices': indices_to_prune.tolist()
                        }

        # Calculate achieved sparsity
        sparsity = self.calculate_sparsity()
        print(f"Achieved sparsity after structured pruning: {sparsity:.4f}")

        return pruned_channels_info

    def gradual_structured_pruning(self, target_sparsity: float, num_steps: int = 5):
        """
        Perform gradual structured pruning.

        Args:
            target_sparsity (float): Target sparsity level
            num_steps (int): Number of pruning steps
        """
        step_ratio = target_sparsity / num_steps

        for step in range(num_steps):
            print(f"Structured pruning step {step + 1}/{num_steps}")
            self.structured_channel_pruning(step_ratio)
            current_sparsity = self.calculate_sparsity()
            print(f"Current sparsity: {current_sparsity:.4f}")

    def importance_based_pruning(self, pruning_ratio: float, importance_scores: Dict[str, torch.Tensor]):
        """
        Perform pruning based on importance scores.

        Args:
            pruning_ratio (float): Fraction of weights to prune
            importance_scores (dict): Dictionary of importance scores for each parameter
        """
        print(f"Performing importance-based pruning with ratio: {pruning_ratio:.2f}")

        # Collect all importance scores
        all_scores = []
        score_info = []

        for name, param in self.model.named_parameters():
            if 'weight' in name and name in importance_scores:
                scores_flat = importance_scores[name].flatten()
                all_scores.append(scores_flat)
                score_info.append((name, param.shape))

        # Concatenate all scores
        all_scores_tensor = torch.cat(all_scores)

        # Find threshold for pruning
        k = int(len(all_scores_tensor) * pruning_ratio)
        if k > 0:
            threshold = torch.topk(all_scores_tensor, k, largest=False)[0][-1]
        else:
            threshold = float('inf')

        # Apply pruning
        for name, param in self.model.named_parameters():
            if 'weight' in name and name in importance_scores:
                mask = (importance_scores[name] > threshold).float()
                param.data.mul_(mask)
                self.masks[name] = mask

        sparsity = self.calculate_sparsity()
        print(f"Achieved sparsity: {sparsity:.4f}")

    def apply_masks(self):
        """Apply stored masks to model parameters."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data.mul_(self.masks[name])

    def remove_pruned_channels_physically(self):
        """
        Physically remove pruned channels from the model architecture.
        This creates a smaller model but requires careful handling of layer connections.
        """
        print("Warning: Physical channel removal not implemented in this version.")
        print("The model retains original architecture with zero weights.")

    def get_pruning_statistics(self) -> Dict:
        """Get detailed pruning statistics."""
        stats = {
            'overall_sparsity': self.calculate_sparsity(),
            'layer_sparsity': {},
            'total_parameters': 0,
            'remaining_parameters': 0
        }

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total = param.numel()
                non_zero = (param.data != 0).sum().item()
                layer_sparsity = 1 - (non_zero / total)

                stats['layer_sparsity'][name] = {
                    'sparsity': layer_sparsity,
                    'total_params': total,
                    'remaining_params': non_zero
                }

                stats['total_parameters'] += total
                stats['remaining_parameters'] += non_zero

        return stats


def calculate_weight_importance_magnitude(model):
    """Calculate importance scores based on weight magnitude."""
    importance_scores = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            importance_scores[name] = param.data.abs()
    return importance_scores


def calculate_weight_importance_gradient(model, data_loader, criterion, device, num_batches=10):
    """
    Calculate importance scores based on gradient information.

    Args:
        model: The neural network model
        data_loader: DataLoader for calculating gradients
        criterion: Loss function
        device: Computing device
        num_batches: Number of batches to use for gradient calculation
    """
    model.eval()
    importance_scores = {}

    # Initialize importance scores
    for name, param in model.named_parameters():
        if 'weight' in name:
            importance_scores[name] = torch.zeros_like(param.data)

    # Calculate gradients over multiple batches
    for i, (inputs, targets) in enumerate(data_loader):
        if i >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Accumulate gradient magnitudes
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                importance_scores[name] += param.grad.abs()

    # Average over batches
    for name in importance_scores:
        importance_scores[name] /= min(num_batches, len(data_loader))

    return importance_scores


