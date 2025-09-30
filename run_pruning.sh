#!/bin/bash
#SBATCH --job-name=pruning
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=pruning_%j.log

module load Python/3.11.3-GCCcore-12.3.0
cd ~/pruning-lab
python main.py --model all --epochs 100
