#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 08:00:00
#SBATCH --constraint=vram32
#SBATCH -o slurm-%j.out  # %j = job ID

module load miniconda/22.11.1-1
conda activate project_5

python3 src/reasoning/main.py \
    --num-examples 200 \
    --num-examples-offset 0 \
    --num-shots 8 \
    --num-samples 5 \
    --temperature 0.9 \
    --model gemma-2b \
    --output-path gemma-2b_gsm8k_T_0.9_samples.json
