#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 7:00:00  # Job time limit
#SBATCH --constraint=vram16
#SBATCH -o slurm-%j.out  # %j = job ID
export PYTHONPATH="${PYTHONPATH}=$(pwd):$PYTHONPATH"

python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=google/gemma-2b-it  --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=5 --eval_dataset_size=500 --eval_output_temperature=0.5 --eval_output_top_k=40 --eval_output_top_p=1 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=random --clean_out_dir=True
python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=google/gemma-2b-it  --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=10 --eval_dataset_size=500 --eval_output_temperature=0.5 --eval_output_top_k=40 --eval_output_top_p=1 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=random --clean_out_dir=True
python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=google/gemma-2b-it  --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=20 --eval_dataset_size=500 --eval_output_temperature=0.5 --eval_output_top_k=40 --eval_output_top_p=1 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=random --clean_out_dir=True
python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=google/gemma-2b-it  --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=5 --eval_dataset_size=500 --eval_output_temperature=0.7 --eval_output_top_k=40 --eval_output_top_p=1 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=random --clean_out_dir=True
python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=google/gemma-2b-it  --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=10 --eval_dataset_size=500 --eval_output_temperature=0.7 --eval_output_top_k=40 --eval_output_top_p=1 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=random --clean_out_dir=True
python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=google/gemma-2b-it  --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=20 --eval_dataset_size=500 --eval_output_temperature=0.7 --eval_output_top_k=40 --eval_output_top_p=1 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=random --clean_out_dir=True
