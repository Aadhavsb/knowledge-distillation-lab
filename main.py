#!/usr/bin/env python3
"""
ECSE 397/600: Efficient Deep Learning - Lab 1
Custom Pruning of ResNet-18 and ViT-Tiny

Main entry point for running experiments.
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from datetime import datetime

# Import our modules
from data.dataloader import get_loaders
from models.resnet18 import get_resnet18_cifar10
from models.vit_tiny import get_vit_tiny_cifar10, get_vit_tiny_pretrained_timm
from train.train_loop import Trainer
from train.prune import CustomPruner
from inference.test import ModelEvaluator


def setup_device():
    """Setup computing device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def train_resnet18(args, device):
    """Train ResNet-18 model."""
    print("\n" + "="*60)
    print("TRAINING RESNET-18")
    print("="*60)

    # Create model
    model = get_resnet18_cifar10(pretrained=args.pretrained).to(device)
    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get data loaders
    train_loader, test_loader = get_loaders(batch_size=args.batch_size)

    # Create trainer
    trainer = Trainer(model, device, save_dir=args.save_dir)

    # Train model
    if not args.skip_training:
        print("Starting training...")
        trainer.train(
            train_loader, test_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            save_best=True
        )

        # Save trained model
        trainer.save_model('cnn_before_pruning.pth')

    # Evaluate trained model
    evaluator = ModelEvaluator(model, device)
    accuracy = evaluator.evaluate_accuracy(test_loader)['overall_accuracy']

    print(f"ResNet-18 training completed. Final accuracy: {accuracy:.2f}%")
    return model, accuracy


def train_vit_tiny(args, device, pretrained=True):
    """Train ViT-Tiny model."""
    model_type = "ViT-Tiny (Pre-trained)" if pretrained else "ViT-Tiny (From Scratch)"
    print(f"\n" + "="*60)
    print(f"TRAINING {model_type.upper()}")
    print("="*60)

    # Create model
    if pretrained:
        try:
            model = get_vit_tiny_pretrained_timm(num_classes=10).to(device)
            print("Using pre-trained ViT-Tiny from timm")
        except ImportError:
            print("timm not available, using custom ViT-Tiny")
            model = get_vit_tiny_cifar10(pretrained=False).to(device)
    else:
        model = get_vit_tiny_cifar10(pretrained=False).to(device)

    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get data loaders
    train_loader, test_loader = get_loaders(batch_size=args.batch_size)

    # Create trainer
    trainer = Trainer(model, device, save_dir=args.save_dir)

    # Train model
    if not args.skip_training:
        print("Starting training...")
        epochs = args.epochs if not pretrained else max(args.epochs // 2, 30)  # Fewer epochs for pre-trained
        lr = args.lr if not pretrained else args.lr * 0.1  # Lower LR for pre-trained

        trainer.train(
            train_loader, test_loader,
            num_epochs=epochs,
            learning_rate=lr,
            save_best=True
        )

        # Save trained model
        trainer.save_model('vit_before_pruning.pth')

    # Evaluate trained model
    evaluator = ModelEvaluator(model, device)
    accuracy = evaluator.evaluate_accuracy(test_loader)['overall_accuracy']

    print(f"{model_type} training completed. Final accuracy: {accuracy:.2f}%")
    return model, accuracy


def perform_pruning_experiment(model, model_name, target_accuracy, args, device):
    """Perform pruning experiments on a model."""
    print(f"\n" + "="*60)
    print(f"PRUNING EXPERIMENTS - {model_name.upper()}")
    print("="*60)

    # Get data loaders
    train_loader, test_loader = get_loaders(batch_size=args.batch_size)

    # Create pruner
    pruner = CustomPruner(model)

    # Save original weights
    pruner.save_original_weights()

    # Get baseline evaluation
    evaluator = ModelEvaluator(model, device)
    baseline_accuracy = evaluator.evaluate_accuracy(test_loader)['overall_accuracy']

    results = {
        'original_accuracy': baseline_accuracy / 100.0
    }

    # Unstructured Pruning
    print(f"\nPerforming unstructured pruning...")
    model_copy = type(model)(**{} if hasattr(model, '__dict__') else {})
    model_copy.load_state_dict(model.state_dict())
    model_copy = model_copy.to(device)

    pruner_unstructured = CustomPruner(model_copy)

    # Try different sparsity levels to find maximum that meets accuracy requirement
    target_sparsities = [0.70, 0.80, 0.85, 0.90, 0.95]
    best_unstructured_sparsity = 0
    best_unstructured_accuracy = 0

    for sparsity in target_sparsities:
        # Reset model
        model_copy.load_state_dict(model.state_dict())
        pruner_unstructured = CustomPruner(model_copy)

        # Apply pruning
        pruner_unstructured.magnitude_based_unstructured_pruning(sparsity)

        # Fine-tune
        trainer = Trainer(model_copy, device, save_dir=args.save_dir)
        trainer.fine_tune(train_loader, test_loader, num_epochs=20, learning_rate=0.01)

        # Evaluate
        evaluator = ModelEvaluator(model_copy, device)
        pruned_accuracy = evaluator.evaluate_accuracy(test_loader, verbose=False)['overall_accuracy']
        achieved_sparsity = pruner_unstructured.calculate_sparsity()

        print(f"Sparsity: {achieved_sparsity:.3f}, Accuracy: {pruned_accuracy:.2f}%")

        if pruned_accuracy >= target_accuracy:
            best_unstructured_sparsity = achieved_sparsity
            best_unstructured_accuracy = pruned_accuracy
            # Save best unstructured model
            torch.save(model_copy.state_dict(),
                      os.path.join(args.save_dir, f'{model_name.lower()}_after_unstructured_pruning.pth'))
        else:
            break

    results['unstructured'] = {
        'pruning_percentage': best_unstructured_sparsity * 100,
        'pruned_accuracy': best_unstructured_accuracy / 100.0
    }

    # Structured Pruning
    print(f"\nPerforming structured pruning...")
    model_copy.load_state_dict(model.state_dict())
    pruner_structured = CustomPruner(model_copy)

    # Try different pruning ratios for structured pruning
    channel_ratios = [0.25, 0.35, 0.45, 0.55, 0.65]
    best_structured_sparsity = 0
    best_structured_accuracy = 0

    for ratio in channel_ratios:
        # Reset model
        model_copy.load_state_dict(model.state_dict())
        pruner_structured = CustomPruner(model_copy)

        # Apply structured pruning
        pruner_structured.structured_channel_pruning(ratio)

        # Fine-tune
        trainer = Trainer(model_copy, device, save_dir=args.save_dir)
        trainer.fine_tune(train_loader, test_loader, num_epochs=20, learning_rate=0.01)

        # Evaluate
        evaluator = ModelEvaluator(model_copy, device)
        pruned_accuracy = evaluator.evaluate_accuracy(test_loader, verbose=False)['overall_accuracy']
        achieved_sparsity = pruner_structured.calculate_sparsity()

        print(f"Channel ratio: {ratio:.2f}, Sparsity: {achieved_sparsity:.3f}, Accuracy: {pruned_accuracy:.2f}%")

        if pruned_accuracy >= target_accuracy:
            best_structured_sparsity = achieved_sparsity
            best_structured_accuracy = pruned_accuracy
            # Save best structured model
            torch.save(model_copy.state_dict(),
                      os.path.join(args.save_dir, f'{model_name.lower()}_after_structured_pruning.pth'))
        else:
            break

    results['structured'] = {
        'pruning_percentage': best_structured_sparsity * 100,
        'pruned_accuracy': best_structured_accuracy / 100.0
    }

    print(f"\nPruning results for {model_name}:")
    print(f"Original accuracy: {baseline_accuracy:.2f}%")
    print(f"Best unstructured: {best_unstructured_sparsity*100:.1f}% sparsity, {best_unstructured_accuracy:.2f}% accuracy")
    print(f"Best structured: {best_structured_sparsity*100:.1f}% sparsity, {best_structured_accuracy:.2f}% accuracy")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Neural Network Pruning Lab')

    # General arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='./models_saved', help='Directory to save models')
    parser.add_argument('--pretrained', action='store_true', help='Use pre-trained weights')
    parser.add_argument('--skip-training', action='store_true', help='Skip training phase')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for computation')

    # Experiment selection
    parser.add_argument('--model', type=str, choices=['resnet18', 'vit', 'all'], default='all',
                       help='Which model to train and prune')
    parser.add_argument('--experiment', type=str, choices=['train', 'prune', 'all'], default='all',
                       help='Which experiment to run')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = setup_device()
    else:
        device = torch.device(args.device)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Results dictionary for report.json
    results = {
        'initial_accuracies': {},
        'unstructured_pruning': {},
        'structured_pruning': {}
    }

    # Train and prune ResNet-18
    if args.model in ['resnet18', 'all']:
        if args.experiment in ['train', 'all']:
            resnet_model, resnet_accuracy = train_resnet18(args, device)
            results['initial_accuracies']['cnn_before_pruning'] = resnet_accuracy / 100.0

        if args.experiment in ['prune', 'all']:
            # Load model if we skipped training
            if args.skip_training:
                resnet_model = get_resnet18_cifar10().to(device)
                resnet_model.load_state_dict(torch.load(
                    os.path.join(args.save_dir, 'cnn_before_pruning.pth'),
                    map_location=device
                ))

            # Target accuracy: 85% for ResNet-18
            resnet_results = perform_pruning_experiment(
                resnet_model, 'cnn', 85.0, args, device
            )

            results['unstructured_pruning']['cnn'] = resnet_results['unstructured']
            results['structured_pruning']['cnn'] = resnet_results['structured']
            results['unstructured_pruning']['cnn']['original_accuracy'] = resnet_results['original_accuracy']
            results['structured_pruning']['cnn']['original_accuracy'] = resnet_results['original_accuracy']

    # Train and prune ViT-Tiny
    if args.model in ['vit', 'all']:
        if args.experiment in ['train', 'all']:
            # Try pre-trained first, then from scratch
            try:
                vit_model, vit_accuracy = train_vit_tiny(args, device, pretrained=True)
                target_vit_accuracy = 88.0  # Higher target for pre-trained
            except Exception as e:
                print(f"Pre-trained ViT failed: {e}")
                print("Falling back to training from scratch...")
                vit_model, vit_accuracy = train_vit_tiny(args, device, pretrained=False)
                target_vit_accuracy = 80.0  # Lower target for from-scratch

            results['initial_accuracies']['vit_before_pruning'] = vit_accuracy / 100.0

        if args.experiment in ['prune', 'all']:
            # Load model if we skipped training
            if args.skip_training:
                vit_model = get_vit_tiny_cifar10().to(device)
                vit_model.load_state_dict(torch.load(
                    os.path.join(args.save_dir, 'vit_before_pruning.pth'),
                    map_location=device
                ))
                target_vit_accuracy = 80.0

            vit_results = perform_pruning_experiment(
                vit_model, 'vit', target_vit_accuracy, args, device
            )

            results['unstructured_pruning']['vit'] = vit_results['unstructured']
            results['structured_pruning']['vit'] = vit_results['structured']
            results['unstructured_pruning']['vit']['original_accuracy'] = vit_results['original_accuracy']
            results['structured_pruning']['vit']['original_accuracy'] = vit_results['original_accuracy']

    # Save results to report.json
    report_path = os.path.join(os.getcwd(), 'report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    print(f"Results saved to: {report_path}")
    print(f"Model checkpoints saved to: {args.save_dir}")

    # Print summary
    print("\nSUMMARY:")
    if 'cnn_before_pruning' in results['initial_accuracies']:
        print(f"ResNet-18 initial accuracy: {results['initial_accuracies']['cnn_before_pruning']:.3f}")
    if 'vit_before_pruning' in results['initial_accuracies']:
        print(f"ViT-Tiny initial accuracy: {results['initial_accuracies']['vit_before_pruning']:.3f}")

    for pruning_type in ['unstructured_pruning', 'structured_pruning']:
        if results[pruning_type]:
            print(f"\n{pruning_type.replace('_', ' ').title()} Results:")
            for model_name, model_results in results[pruning_type].items():
                sparsity = model_results.get('pruning_percentage', 0)
                accuracy = model_results.get('pruned_accuracy', 0)
                print(f"  {model_name}: {sparsity:.1f}% sparsity, {accuracy:.3f} accuracy")


if __name__ == '__main__':
    main()