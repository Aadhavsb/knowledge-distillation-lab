#!/bin/bash
#SBATCH --job-name=lab2_kd
#SBATCH --output=lab2_output_%j.log
#SBATCH --error=lab2_error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=06:00:00

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load required modules (using correct names from your cluster)
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo ""
echo "Loaded modules:"
module list

# Print GPU information
echo ""
nvidia-smi

# Print Python and PyTorch versions
echo ""
echo "Python version:"
python --version

echo ""
echo "PyTorch information:"
python -s -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')"

echo ""
echo "Torchvision check:"
python -s -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

echo ""
echo "=========================================="
echo "Starting Knowledge Distillation Training"
echo "=========================================="

# Run the main training script with -s flag to ignore user site-packages
python -s main.py \
    --teacher-epochs 100 \
    --student-epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --alpha 0.5 \
    --temperature 4.0 \
    --save-dir ./models_saved

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="

    # Display the final report
    echo ""
    echo "Final Results (report.json):"
    cat report.json

    # List saved models
    echo ""
    echo "Saved model checkpoints:"
    ls -lh models_saved/*.pth

else
    echo ""
    echo "=========================================="
    echo "Training failed with error code $?"
    echo "=========================================="
    exit 1
fi

echo ""
echo "End Time: $(date)"
echo "=========================================="
