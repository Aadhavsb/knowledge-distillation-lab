#!/bin/bash
#SBATCH --job-name=pruning
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH -t 05:00:00
#SBATCH --output=pruning_%j.log

module load Python/3.11.3-GCCcore-12.3.0
cd ~/pruning-lab
python main.py --model all --epochs 100
