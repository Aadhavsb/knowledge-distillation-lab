#!/bin/bash
#SBATCH --job-name=pruning
#SBATCH -p gpu
#SBATCH -c 24
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB
#SBATCH -t 24:00:00
#SBATCH --output=pruning_%j.log

module load Python/3.11.3-GCCcore-12.3.0
cd ~/pruning-lab
python main.py --model all --epochs 100
