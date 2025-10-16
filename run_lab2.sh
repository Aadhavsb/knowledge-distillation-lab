#!/bin/bash
#SBATCH --job-name=lab2_kd
#SBATCH --output=lab2_output_%j.log
#SBATCH --error=lab2_error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@case.edu

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load required modules (adjust based on your cluster)
module load python/3.9
module load cuda/11.8

# Print GPU information
nvidia-smi

# Activate virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Install required packages if not already installed
pip install torch torchvision tqdm --quiet

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Print Python and PyTorch versions
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

echo "=========================================="
echo "Starting Knowledge Distillation Training"
echo "=========================================="

# Run the main training script
# Using 100 epochs for both teachers and students as per assignment
# Alpha=0.5 and Temperature=4.0 are good defaults for KD
python main.py \
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
