#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 16:00:00  # Job time limit
#SBATCH --constraint=vram48
#SBATCH -o slurm-%j.out  # %j = job ID
export PYTHONPATH="${PYTHONPATH}=$(pwd):$PYTHONPATH"

python3 src/self_consistency_merge.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=meta-llama/Llama-2-7b-hf --load_in_8bit --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=gsm8k --dataset_subname=main --eval_n_samples=40 --eval_dataset_size=500 --eval_dataset_range=0:500    --eval_output_temperature=1 --eval_output_top_k=40 --eval_output_top_p=0.95 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=static --clean_out_dir=True
python3 src/self_consistency_merge.py --out_dir=outputs/self_consistency --debug --verbose --seed=42 --run_id=0 --model=meta-llama/Llama-2-7b-hf --load_in_8bit --eval_inf_fn_key=fewshot --eval_split=validation --dataset_name=gsm8k --dataset_subname=main --eval_n_samples=40 --eval_dataset_size=500 --eval_dataset_range=500:1000 --eval_output_temperature=1 --eval_output_top_k=40 --eval_output_top_p=0.95 --eval_retrieval_strategy=random --eval_output_sampling_strategy=arithmetic --eval_output_beam_size=1 --dataset_sample_strategy=static --clean_out_dir=True
