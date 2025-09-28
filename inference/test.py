import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple


class ModelEvaluator:
    """Comprehensive model evaluation for pruned networks."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate_accuracy(self, test_loader, verbose=True):
        """
        Evaluate model accuracy on test set.

        Args:
            test_loader: Test data loader
            verbose: Whether to print detailed results

        Returns:
            Dictionary with accuracy metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))  # CIFAR-10 has 10 classes
        class_total = list(0. for i in range(10))

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # Per-class accuracy
                c = (predicted == targets).squeeze()
                for i in range(targets.size(0)):
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        overall_accuracy = 100.0 * correct / total

        # Calculate per-class accuracies
        class_accuracies = []
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)

        results = {
            'overall_accuracy': overall_accuracy,
            'correct_predictions': correct,
            'total_samples': total,
            'class_accuracies': class_accuracies,
            'mean_class_accuracy': np.mean(class_accuracies)
        }

        if verbose:
            print(f"Overall Accuracy: {overall_accuracy:.2f}% ({correct}/{total})")
            print(f"Mean Class Accuracy: {np.mean(class_accuracies):.2f}%")

            class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            print("\nPer-class accuracies:")
            for i, (class_name, acc) in enumerate(zip(class_names, class_accuracies)):
                print(f"  {class_name}: {acc:.2f}%")

        return results

    def measure_inference_time(self, test_loader, num_batches=100, warmup_batches=10):
        """
        Measure inference time performance.

        Args:
            test_loader: Test data loader
            num_batches: Number of batches to measure
            warmup_batches: Number of warmup batches

        Returns:
            Dictionary with timing metrics
        """
        self.model.eval()

        # Warmup
        print(f"Warming up for {warmup_batches} batches...")
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                if i >= warmup_batches:
                    break
                inputs = inputs.to(self.device)
                _ = self.model(inputs)

        # Actual measurement
        print(f"Measuring inference time for {num_batches} batches...")
        times = []
        samples_processed = 0

        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                if i >= num_batches:
                    break

                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)

                # Synchronize CUDA operations
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                start_time = time.time()
                _ = self.model(inputs)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.time()

                batch_time = end_time - start_time
                times.append(batch_time)
                samples_processed += batch_size

        avg_batch_time = np.mean(times)
        std_batch_time = np.std(times)
        avg_sample_time = avg_batch_time / (samples_processed / len(times))

        results = {
            'avg_batch_time_ms': avg_batch_time * 1000,
            'std_batch_time_ms': std_batch_time * 1000,
            'avg_sample_time_ms': avg_sample_time * 1000,
            'throughput_samples_per_sec': 1.0 / avg_sample_time,
            'total_batches_measured': len(times),
            'total_samples_processed': samples_processed
        }

        print(f"Average batch time: {avg_batch_time * 1000:.2f} ± {std_batch_time * 1000:.2f} ms")
        print(f"Average per-sample time: {avg_sample_time * 1000:.2f} ms")
        print(f"Throughput: {1.0 / avg_sample_time:.1f} samples/sec")

        return results

    def analyze_model_sparsity(self):
        """
        Analyze sparsity statistics of the model.

        Returns:
            Dictionary with sparsity metrics
        """
        total_params = 0
        zero_params = 0
        layer_stats = {}

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                layer_total = param.numel()
                layer_zeros = (param.data == 0).sum().item()
                layer_sparsity = layer_zeros / layer_total

                layer_stats[name] = {
                    'total_params': layer_total,
                    'zero_params': layer_zeros,
                    'sparsity': layer_sparsity,
                    'density': 1 - layer_sparsity
                }

                total_params += layer_total
                zero_params += layer_zeros

        overall_sparsity = zero_params / total_params if total_params > 0 else 0

        results = {
            'overall_sparsity': overall_sparsity,
            'overall_density': 1 - overall_sparsity,
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'remaining_parameters': total_params - zero_params,
            'layer_statistics': layer_stats
        }

        print(f"Overall Model Sparsity: {overall_sparsity:.4f} ({overall_sparsity * 100:.2f}%)")
        print(f"Total Parameters: {total_params:,}")
        print(f"Zero Parameters: {zero_params:,}")
        print(f"Remaining Parameters: {total_params - zero_params:,}")

        return results

    def calculate_model_size(self):
        """Calculate model size in different units."""
        param_size = 0
        buffer_size = 0

        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size_bytes = param_size + buffer_size
        size_mb = total_size_bytes / (1024 ** 2)
        size_kb = total_size_bytes / 1024

        results = {
            'size_bytes': total_size_bytes,
            'size_kb': size_kb,
            'size_mb': size_mb,
            'param_size_bytes': param_size,
            'buffer_size_bytes': buffer_size
        }

        print(f"Model Size: {size_mb:.2f} MB ({size_kb:.1f} KB)")

        return results

    def comprehensive_evaluation(self, test_loader, measure_timing=True):
        """
        Perform comprehensive evaluation of the model.

        Args:
            test_loader: Test data loader
            measure_timing: Whether to measure inference timing

        Returns:
            Dictionary with all evaluation metrics
        """
        print("=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)

        results = {}

        # Accuracy evaluation
        print("\n1. ACCURACY EVALUATION")
        print("-" * 30)
        results['accuracy'] = self.evaluate_accuracy(test_loader)

        # Sparsity analysis
        print("\n2. SPARSITY ANALYSIS")
        print("-" * 30)
        results['sparsity'] = self.analyze_model_sparsity()

        # Model size
        print("\n3. MODEL SIZE")
        print("-" * 30)
        results['size'] = self.calculate_model_size()

        # Timing analysis (optional)
        if measure_timing:
            print("\n4. INFERENCE TIMING")
            print("-" * 30)
            results['timing'] = self.measure_inference_time(test_loader)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        return results

    def compare_with_baseline(self, baseline_results, current_results):
        """
        Compare current model with baseline model.

        Args:
            baseline_results: Results from baseline model evaluation
            current_results: Results from current model evaluation

        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}

        # Accuracy comparison
        baseline_acc = baseline_results['accuracy']['overall_accuracy']
        current_acc = current_results['accuracy']['overall_accuracy']
        acc_drop = baseline_acc - current_acc
        acc_retention = current_acc / baseline_acc

        comparison['accuracy'] = {
            'baseline_accuracy': baseline_acc,
            'current_accuracy': current_acc,
            'accuracy_drop': acc_drop,
            'accuracy_retention': acc_retention
        }

        # Size comparison
        baseline_size = baseline_results['size']['size_mb']
        current_size = current_results['size']['size_mb']
        size_reduction = (baseline_size - current_size) / baseline_size
        compression_ratio = baseline_size / current_size

        comparison['size'] = {
            'baseline_size_mb': baseline_size,
            'current_size_mb': current_size,
            'size_reduction': size_reduction,
            'compression_ratio': compression_ratio
        }

        # Sparsity comparison
        baseline_sparsity = baseline_results['sparsity']['overall_sparsity']
        current_sparsity = current_results['sparsity']['overall_sparsity']

        comparison['sparsity'] = {
            'baseline_sparsity': baseline_sparsity,
            'current_sparsity': current_sparsity,
            'sparsity_increase': current_sparsity - baseline_sparsity
        }

        print(f"Accuracy: {baseline_acc:.2f}% → {current_acc:.2f}% (drop: {acc_drop:.2f}%)")
        print(f"Size: {baseline_size:.2f} MB → {current_size:.2f} MB (reduction: {size_reduction * 100:.1f}%)")
        print(f"Sparsity: {baseline_sparsity:.4f} → {current_sparsity:.4f}")

        return comparison


