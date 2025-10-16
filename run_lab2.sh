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

# Try to load modules if they exist (ignore errors)
module load python 2>/dev/null || true
module load cuda 2>/dev/null || true
module load anaconda3 2>/dev/null || true

# Print GPU information
nvidia-smi

# Find python (try multiple options)
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "ERROR: No Python found!"
    exit 1
fi

echo ""
echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Check if PyTorch is installed
if $PYTHON_CMD -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed"
    $PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
else
    echo "Installing PyTorch..."
    $PYTHON_CMD -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
fi

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

echo "=========================================="
echo "Starting Knowledge Distillation Training"
echo "=========================================="

# Run the main training script
$PYTHON_CMD main.py \
    --teacher-epochs 100 \
    --student-epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --alpha 0.5 \
    --temperature 4.0 \
    --save-dir ./models_saved

# Check exit status
if [ $? -eq 0 ]; then
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
    echo "=========================================="
    echo "Training failed with error code $?"
    echo "=========================================="
    exit 1
fi

echo ""
echo "End Time: $(date)"
echo "=========================================="
