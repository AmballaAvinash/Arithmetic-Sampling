#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2
#SBATCH --mem=50G
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 6:00:00
#SBATCH --constraint=vram23
#SBATCH -o slurm-%j.out
python3 src/reasoning/main.py
