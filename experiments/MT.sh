#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 30:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --constraint=[a100]
export PYTHONPATH="${PYTHONPATH}=$(pwd):$PYTHONPATH"


module load miniconda/22.11.1-1

conda activate 696_ds

python src/test_de_en1.py