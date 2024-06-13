#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 8:00:00  # Job time limit
#SBATCH --constraint=vram80
#SBATCH -o slurm-%j.out  # %j = job ID
export PYTHONPATH="${PYTHONPATH}=$(pwd):$PYTHONPATH"
#temp=0.5,k=40
#nsamples 5 temp 0.5
#temp=0.5,k=40
python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=meta-llama/Llama-2-7b-hf  --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=20 --eval_dataset_size=2300 --eval_dataset_range=0:2300 --eval_output_temperature=1 --eval_output_top_k=40 --eval_output_top_p=1 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=static --clean_out_dir=True
# python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=meta-llama/Llama-2-7b-hf  --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=20 --eval_dataset_size=500  --eval_dataset_range=500:1000 --eval_output_temperature=1 --eval_output_top_k=40 --eval_output_top_p=0.95 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=static --clean_out_dir=True
# python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=meta-llama/Llama-2-7b-hf  --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=20 --eval_dataset_size=500 --eval_dataset_range=1000:1500 --eval_output_temperature=1 --eval_output_top_k=40 --eval_output_top_p=0.95 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=static --clean_out_dir=True
# python3 src/self_consistency.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=meta-llama/Llama-2-7b-hf  --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=strategy_qa --dataset_subname=None --eval_n_samples=20 --eval_dataset_size=500  --eval_dataset_range=1500:2000 --eval_output_temperature=1 --eval_output_top_k=40 --eval_output_top_p=0.95 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=static --clean_out_dir=True


