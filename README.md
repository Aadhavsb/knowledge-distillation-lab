# Lab 2: Knowledge Distillation with ResNet-18 and ViT-Tiny

ECSE 397/600: Efficient Deep Learning

## Overview

This lab implements knowledge distillation (KD) to train compact student models guided by larger teacher models on CIFAR-10.

## Project Structure

```
distillation_lab/
├── data/
│   └── dataloader.py          # CIFAR-10 data loading with augmentations
├── models/
│   ├── teacher_resnet.py      # ResNet-18 teacher (2 blocks per stage)
│   ├── teacher_vit.py         # ViT-Tiny teacher (12 layers, 192 dim)
│   ├── student_resnet.py      # ResNet-8 student (1 block per stage)
│   └── student_vit.py         # ViT student (6 layers, 192 dim)
├── train/
│   ├── train_teacher.py       # Teacher training module
│   └── distill.py             # KD training module
├── inference/
│   └── test.py                # Model evaluation
├── utils/
│   └── kd_losses.py           # KD loss implementations
├── main.py                    # Main training pipeline
├── run_lab2.sh                # SLURM batch script
└── report.json                # Results (generated after training)
```

## Running on GPU Cluster

### Quick Start

1. **Submit the job:**
   ```bash
   sbatch run_lab2.sh
   ```

2. **Monitor the job:**
   ```bash
   squeue -u $USER
   tail -f lab2_output_*.log
   ```

### Manual Run

If you prefer to run interactively:

```bash
# Request interactive GPU session
salloc -p gpu -c 8 --gres=gpu:1 --mem=32gb -t 6:00:00

# Load modules
module load python/3.9 cuda/11.8

# Run training
python main.py --teacher-epochs 100 --student-epochs 100
```

## Command Line Arguments

```bash
python main.py [OPTIONS]

Options:
  --batch-size INT            Batch size (default: 128)
  --teacher-epochs INT        Teacher training epochs (default: 100)
  --student-epochs INT        Student training epochs (default: 100)
  --lr FLOAT                  Learning rate (default: 0.1)
  --alpha FLOAT              KD loss weight (default: 0.5)
  --temperature FLOAT         KD temperature (default: 4.0)
  --save-dir PATH             Model save directory (default: ./models_saved)

  # Skip flags (for resuming training)
  --skip-cnn-teacher          Skip CNN teacher training
  --skip-vit-teacher          Skip ViT teacher training
  --skip-cnn-student-no-kd    Skip CNN student baseline
  --skip-vit-student-no-kd    Skip ViT student baseline
  --skip-cnn-student-kd       Skip CNN student with KD
  --skip-vit-student-kd       Skip ViT student with KD
```

## Expected Training Time

On a single GPU (RTX 2080 Ti or similar):
- **Total**: ~4-5 hours for complete pipeline
- ResNet-18 Teacher: ~35 min
- ViT-Tiny Teacher: ~75 min
- ResNet-8 Student (no KD): ~25 min
- ViT Student (no KD): ~40 min
- ResNet-8 Student (with KD): ~30 min
- ViT Student (with KD): ~45 min

## Output Files

After training completes, you'll have:

### Model Checkpoints (models_saved/)
- `cnn_teacher.pth` - ResNet-18 teacher
- `cnn_student_no_kd.pth` - ResNet-8 baseline
- `cnn_student_with_kd.pth` - ResNet-8 with KD
- `vit_teacher.pth` - ViT-Tiny teacher
- `vit_student_no_kd.pth` - ViT student baseline
- `vit_student_with_kd.pth` - ViT student with KD

### Results
- `report.json` - Accuracy metrics in required format

Example report.json:
```json
{
  "cnn": {
    "teacher_accuracy": 0.912,
    "student_accuracy_without_kd": 0.845,
    "student_accuracy_with_kd": 0.872
  },
  "vit": {
    "teacher_accuracy": 0.927,
    "student_accuracy_without_kd": 0.812,
    "student_accuracy_with_kd": 0.854
  }
}
```

## Knowledge Distillation Loss

The KD loss combines hard targets (true labels) and soft targets (teacher predictions):

```
L_KD = α · L_CE(y, p_s) + (1 - α) · T² · KL(p_t^T || p_s^T)
```

Where:
- `α` = weight for hard target loss (default: 0.5)
- `T` = temperature for softening distributions (default: 4.0)
- `L_CE` = cross-entropy loss with true labels
- `KL` = KL divergence between teacher and student distributions

## Model Architectures

### Teachers
- **ResNet-18**: 11.2M parameters, [2,2,2,2] blocks
- **ViT-Tiny**: 5.7M parameters, 12 layers, 192 dim

### Students
- **ResNet-8**: 5.6M parameters, [1,1,1,1] blocks (50% fewer layers)
- **ViT-Student**: 2.9M parameters, 6 layers, 192 dim (50% fewer layers)

## Requirements

```
torch>=1.12.0
torchvision>=0.13.0
```

Install with:
```bash
pip install torch torchvision
```

## Troubleshooting

**Out of memory errors:**
- Reduce `--batch-size` to 64 or 32
- Check GPU memory: `nvidia-smi`

**Training too slow:**
- Verify GPU is being used: check log for "Using GPU"
- Reduce epochs for testing: `--teacher-epochs 20 --student-epochs 20`

**Resume from checkpoint:**
Use skip flags to resume training from a specific point.

## References

- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- PyTorch KD example: https://github.com/peterliht/knowledge-distillation-pytorch
- ResNet-CIFAR10: https://github.com/akamaster/pytorch_resnet_cifar10
