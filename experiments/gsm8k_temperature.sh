#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 6:00:00
#SBATCH --constraint=vram40
#SBATCH -o slurm-%j.out  # %j = job ID

python3 src/reasoning/main.py \
    --num-examples 200 \
    --num-shots 0 \
    --num-samples 20 \
    --temperature 0.3 0.5 0.7 \
    --top-k 40 \
    --model gemma-7b-it \
    --quant-8-bit \
    > gemma_7b_it-gsm8k-eval_temperatures.txt

python3 src/reasoning/main.py \
    --num-examples 200 \
    --num-shots 0 \
    --num-samples 20 \
    --top-p 0.95 0.9 \
    --model gemma-7b-it \
    --quant-8-bit \
    > gemma_7b_it-gsm8k-eval_topP.txt

python3 src/reasoning/main.py \
    --num-examples 200 \
    --num-shots 0 \
    --num-samples 20 \
    --temperature 0.7 \
    --top-k 20 50 \
    --model gemma-7b-it \
    --quant-8-bit \
    > gemma_7b_it-gsm8k-eval_topK.txt
