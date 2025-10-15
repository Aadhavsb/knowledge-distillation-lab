#!/usr/bin/env python3
"""
ECSE 397/600: Efficient Deep Learning - Lab 2
Knowledge Distillation with ResNet-18 and ViT-Tiny

Main entry point for running experiments.
"""

import argparse
import json
import os
import torch

# Import our modules
from data.dataloader import get_loaders
from models.teacher_resnet import get_resnet18_teacher
from models.teacher_vit import get_vit_tiny_teacher
from models.student_resnet import get_resnet8_student
from models.student_vit import get_vit_student
from train.train_teacher import train_teacher_model
from train.distill import train_student_with_kd, train_student_without_kd
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


def train_teachers(args, device, train_loader, test_loader):
    """Train both teacher models."""
    results = {}

    print("\n" + "="*80)
    print("STEP 1: TRAINING TEACHER MODELS")
    print("="*80)

    # Train ResNet-18 Teacher
    if args.skip_cnn_teacher:
        print("\nSkipping CNN teacher training (loading from checkpoint)...")
        cnn_teacher = get_resnet18_teacher().to(device)
        cnn_teacher.load_state_dict(torch.load(
            os.path.join(args.save_dir, 'cnn_teacher.pth'),
            map_location=device
        ))
        evaluator = ModelEvaluator(cnn_teacher, device)
        cnn_teacher_acc = evaluator.evaluate_accuracy(test_loader)['overall_accuracy']
    else:
        print("\nTraining ResNet-18 Teacher...")
        cnn_teacher = get_resnet18_teacher().to(device)
        _, cnn_teacher_acc = train_teacher_model(
            cnn_teacher, train_loader, test_loader, device,
            model_name="cnn_teacher",
            save_dir=args.save_dir,
            num_epochs=args.teacher_epochs,
            learning_rate=args.lr
        )

    results['cnn_teacher_accuracy'] = cnn_teacher_acc / 100.0
    print(f"\nResNet-18 Teacher Final Accuracy: {cnn_teacher_acc:.2f}%")

    # Train ViT-Tiny Teacher
    if args.skip_vit_teacher:
        print("\nSkipping ViT teacher training (loading from checkpoint)...")
        vit_teacher = get_vit_tiny_teacher().to(device)
        vit_teacher.load_state_dict(torch.load(
            os.path.join(args.save_dir, 'vit_teacher.pth'),
            map_location=device
        ))
        evaluator = ModelEvaluator(vit_teacher, device)
        vit_teacher_acc = evaluator.evaluate_accuracy(test_loader)['overall_accuracy']
    else:
        print("\nTraining ViT-Tiny Teacher...")
        vit_teacher = get_vit_tiny_teacher().to(device)
        _, vit_teacher_acc = train_teacher_model(
            vit_teacher, train_loader, test_loader, device,
            model_name="vit_teacher",
            save_dir=args.save_dir,
            num_epochs=args.teacher_epochs,
            learning_rate=args.lr
        )

    results['vit_teacher_accuracy'] = vit_teacher_acc / 100.0
    print(f"\nViT-Tiny Teacher Final Accuracy: {vit_teacher_acc:.2f}%")

    return cnn_teacher, vit_teacher, results


def train_students_without_kd(args, device, train_loader, test_loader):
    """Train student models without knowledge distillation (baseline)."""
    results = {}

    print("\n" + "="*80)
    print("STEP 2: TRAINING STUDENT MODELS WITHOUT KD (BASELINE)")
    print("="*80)

    # Train ResNet-8 Student without KD
    if args.skip_cnn_student_no_kd:
        print("\nSkipping CNN student (no KD) training (loading from checkpoint)...")
        cnn_student = get_resnet8_student().to(device)
        cnn_student.load_state_dict(torch.load(
            os.path.join(args.save_dir, 'cnn_student_no_kd.pth'),
            map_location=device
        ))
        evaluator = ModelEvaluator(cnn_student, device)
        cnn_student_no_kd_acc = evaluator.evaluate_accuracy(test_loader)['overall_accuracy']
    else:
        print("\nTraining ResNet-8 Student (without KD)...")
        cnn_student = get_resnet8_student().to(device)
        _, cnn_student_no_kd_acc = train_student_without_kd(
            cnn_student, train_loader, test_loader, device,
            model_name="cnn_student_no_kd",
            save_dir=args.save_dir,
            num_epochs=args.student_epochs,
            learning_rate=args.lr
        )

    results['cnn_student_no_kd_accuracy'] = cnn_student_no_kd_acc / 100.0
    print(f"\nResNet-8 Student (no KD) Final Accuracy: {cnn_student_no_kd_acc:.2f}%")

    # Train ViT Student without KD
    if args.skip_vit_student_no_kd:
        print("\nSkipping ViT student (no KD) training (loading from checkpoint)...")
        vit_student = get_vit_student().to(device)
        vit_student.load_state_dict(torch.load(
            os.path.join(args.save_dir, 'vit_student_no_kd.pth'),
            map_location=device
        ))
        evaluator = ModelEvaluator(vit_student, device)
        vit_student_no_kd_acc = evaluator.evaluate_accuracy(test_loader)['overall_accuracy']
    else:
        print("\nTraining ViT Student (without KD)...")
        vit_student = get_vit_student().to(device)
        _, vit_student_no_kd_acc = train_student_without_kd(
            vit_student, train_loader, test_loader, device,
            model_name="vit_student_no_kd",
            save_dir=args.save_dir,
            num_epochs=args.student_epochs,
            learning_rate=args.lr
        )

    results['vit_student_no_kd_accuracy'] = vit_student_no_kd_acc / 100.0
    print(f"\nViT Student (no KD) Final Accuracy: {vit_student_no_kd_acc:.2f}%")

    return results


def train_students_with_kd(args, device, cnn_teacher, vit_teacher,
                           train_loader, test_loader):
    """Train student models with knowledge distillation."""
    results = {}

    print("\n" + "="*80)
    print("STEP 3: TRAINING STUDENT MODELS WITH KD")
    print("="*80)

    # Train ResNet-8 Student with KD
    if args.skip_cnn_student_kd:
        print("\nSkipping CNN student (with KD) training (loading from checkpoint)...")
        cnn_student_kd = get_resnet8_student().to(device)
        cnn_student_kd.load_state_dict(torch.load(
            os.path.join(args.save_dir, 'cnn_student_with_kd.pth'),
            map_location=device
        ))
        evaluator = ModelEvaluator(cnn_student_kd, device)
        cnn_student_kd_acc = evaluator.evaluate_accuracy(test_loader)['overall_accuracy']
    else:
        print("\nTraining ResNet-8 Student (with KD)...")
        cnn_student_kd = get_resnet8_student().to(device)
        _, cnn_student_kd_acc = train_student_with_kd(
            cnn_student_kd, cnn_teacher, train_loader, test_loader, device,
            model_name="cnn_student_with_kd",
            save_dir=args.save_dir,
            num_epochs=args.student_epochs,
            learning_rate=args.lr,
            alpha=args.alpha,
            temperature=args.temperature
        )

    results['cnn_student_kd_accuracy'] = cnn_student_kd_acc / 100.0
    print(f"\nResNet-8 Student (with KD) Final Accuracy: {cnn_student_kd_acc:.2f}%")

    # Train ViT Student with KD
    if args.skip_vit_student_kd:
        print("\nSkipping ViT student (with KD) training (loading from checkpoint)...")
        vit_student_kd = get_vit_student().to(device)
        vit_student_kd.load_state_dict(torch.load(
            os.path.join(args.save_dir, 'vit_student_with_kd.pth'),
            map_location=device
        ))
        evaluator = ModelEvaluator(vit_student_kd, device)
        vit_student_kd_acc = evaluator.evaluate_accuracy(test_loader)['overall_accuracy']
    else:
        print("\nTraining ViT Student (with KD)...")
        vit_student_kd = get_vit_student().to(device)
        _, vit_student_kd_acc = train_student_with_kd(
            vit_student_kd, vit_teacher, train_loader, test_loader, device,
            model_name="vit_student_with_kd",
            save_dir=args.save_dir,
            num_epochs=args.student_epochs,
            learning_rate=args.lr,
            alpha=args.alpha,
            temperature=args.temperature
        )

    results['vit_student_kd_accuracy'] = vit_student_kd_acc / 100.0
    print(f"\nViT Student (with KD) Final Accuracy: {vit_student_kd_acc:.2f}%")

    return results


def generate_report(teacher_results, student_no_kd_results, student_kd_results):
    """Generate report.json with all results."""
    report = {
        "cnn": {
            "teacher_accuracy": teacher_results['cnn_teacher_accuracy'],
            "student_accuracy_without_kd": student_no_kd_results['cnn_student_no_kd_accuracy'],
            "student_accuracy_with_kd": student_kd_results['cnn_student_kd_accuracy']
        },
        "vit": {
            "teacher_accuracy": teacher_results['vit_teacher_accuracy'],
            "student_accuracy_without_kd": student_no_kd_results['vit_student_no_kd_accuracy'],
            "student_accuracy_with_kd": student_kd_results['vit_student_kd_accuracy']
        }
    }

    return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Knowledge Distillation Lab')

    # General arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--teacher-epochs', type=int, default=100, help='Number of epochs for teacher training')
    parser.add_argument('--student-epochs', type=int, default=100, help='Number of epochs for student training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='./models_saved', help='Directory to save models')

    # KD parameters
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for hard target loss (0 to 1)')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for soft targets')

    # Skip flags for faster testing/resume
    parser.add_argument('--skip-cnn-teacher', action='store_true', help='Skip CNN teacher training')
    parser.add_argument('--skip-vit-teacher', action='store_true', help='Skip ViT teacher training')
    parser.add_argument('--skip-cnn-student-no-kd', action='store_true', help='Skip CNN student (no KD) training')
    parser.add_argument('--skip-vit-student-no-kd', action='store_true', help='Skip ViT student (no KD) training')
    parser.add_argument('--skip-cnn-student-kd', action='store_true', help='Skip CNN student (with KD) training')
    parser.add_argument('--skip-vit-student-kd', action='store_true', help='Skip ViT student (with KD) training')

    args = parser.parse_args()

    # Setup device
    device = setup_device()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Get data loaders
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_loaders(batch_size=args.batch_size)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Train teachers
    cnn_teacher, vit_teacher, teacher_results = train_teachers(
        args, device, train_loader, test_loader
    )

    # Train students without KD
    student_no_kd_results = train_students_without_kd(
        args, device, train_loader, test_loader
    )

    # Train students with KD
    student_kd_results = train_students_with_kd(
        args, device, cnn_teacher, vit_teacher, train_loader, test_loader
    )

    # Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)

    report = generate_report(teacher_results, student_no_kd_results, student_kd_results)

    report_path = os.path.join(os.getcwd(), 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    print(f"Model checkpoints saved to: {args.save_dir}")

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print("\nCNN (ResNet):")
    print(f"  Teacher Accuracy:              {report['cnn']['teacher_accuracy']:.4f} ({report['cnn']['teacher_accuracy']*100:.2f}%)")
    print(f"  Student Accuracy (without KD): {report['cnn']['student_accuracy_without_kd']:.4f} ({report['cnn']['student_accuracy_without_kd']*100:.2f}%)")
    print(f"  Student Accuracy (with KD):    {report['cnn']['student_accuracy_with_kd']:.4f} ({report['cnn']['student_accuracy_with_kd']*100:.2f}%)")
    print(f"  KD Improvement:                +{(report['cnn']['student_accuracy_with_kd'] - report['cnn']['student_accuracy_without_kd'])*100:.2f}%")

    print("\nViT (Vision Transformer):")
    print(f"  Teacher Accuracy:              {report['vit']['teacher_accuracy']:.4f} ({report['vit']['teacher_accuracy']*100:.2f}%)")
    print(f"  Student Accuracy (without KD): {report['vit']['student_accuracy_without_kd']:.4f} ({report['vit']['student_accuracy_without_kd']*100:.2f}%)")
    print(f"  Student Accuracy (with KD):    {report['vit']['student_accuracy_with_kd']:.4f} ({report['vit']['student_accuracy_with_kd']*100:.2f}%)")
    print(f"  KD Improvement:                +{(report['vit']['student_accuracy_with_kd'] - report['vit']['student_accuracy_without_kd'])*100:.2f}%")

    print("\n" + "="*80)
    print("LAB 2 COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == '__main__':
    main()
