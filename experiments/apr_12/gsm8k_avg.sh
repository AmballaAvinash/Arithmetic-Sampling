#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 4:00:00
#SBATCH --constraint=vram23
#SBATCH -o slurm-%j.out  # %j = job ID

module load miniconda/22.11.1-1
conda activate project_5

python3 src/reasoning/main.py \
    --num-examples 100 \
    --num-shots 0 \
    --num-samples 20 \
    --temperature 0.5 \
    --top-k 40 \
    --model gemma-2b-it \
    --seed 786 275 606 402 702 \
    > gemma_2b_it-gsm8k-eval_multi_sample.txt
