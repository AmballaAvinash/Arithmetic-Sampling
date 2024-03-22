#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 6:00:00  # Job time limit
#SBATCH --constraint=vram23
#SBATCH -o slurm-%j.out  # %j = job ID
export PYTHONPATH="${PYTHONPATH}=$(pwd):$PYTHONPATH"
python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=meta-llama/Llama-2-13b-hf --load_in_8bit  --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=2 --eval_retrieval_strategy=random --eval_output_sampling_strategy=all --eval_output_beam_size=1 --dataset_sample_strategy=random --clean_out_dir=True
python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=google/gemma-2b   --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=250 --eval_retrieval_strategy=random --eval_output_sampling_strategy=all --eval_output_beam_size=1 --dataset_sample_strategy=random --clean_out_dir=True
python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=google/gemma-2b  --load_in_8bit --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=2 --eval_retrieval_strategy=random --eval_output_sampling_strategy=all --eval_output_beam_size=1 --dataset_sample_strategy=random --clean_out_dir=True
