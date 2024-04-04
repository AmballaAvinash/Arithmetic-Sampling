import os
import json
import argparse
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm
from utils import *

# Reference: https://github.com/kojima-takeshi188/zero_shot_cot/blob/main/main.py

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cot_reasoning(
    output_filename, model, tokenizer, data, prompt_prefix, sampling_strategies, num_samples, temperature, num_examples, num_shots, top_k, top_p
):
    arg_key = f"num_examples={num_examples}, num_samples={num_samples}, temperature={temperature}, shots={num_shots}, top_k={top_k}, top_p={top_p}"
    outputs = {}
    if os.path.exists(output_filename):
        with open(output_filename, "r") as output_file:
            outputs = json.load(output_file)
    outputs[arg_key] = []
    for example in tqdm(data, desc="Predicting"):
        question, answer = example["question"], example["answer"]
        answer = answer.split("####")[-1].strip()
        input_prompt = f"Question: {question} \nAnswer: {COT_TRIGGER}"
        if len(prompt_prefix):
            input_prompt = prompt_prefix + input_prompt
        # input_ids = tokenizer(
        #     input_prompt, truncation=True, return_tensors="pt"
        # ).input_ids.to(DEVICE)
        input_ids = tokenizer(
            input_prompt, truncation=True, return_tensors="pt"
        ).input_ids
        if "arithmetic" in sampling_strategies:
            arithmetic_sample_outputs = model.generate(
                input_ids=input_ids,
                num_return_sequences=num_samples,
                do_sample=True,
                max_new_tokens=150,
                temperature=temperature,
                num_beams=1,
                top_k=50 if top_k is None else top_k,
                top_p=1 if top_p is None else top_p, 
                use_arithmetic=True,
            )
        if "ancestral" in sampling_strategies:
            ancestral_sample_outputs = model.generate(
                input_ids=input_ids,
                num_return_sequences=num_samples,
                do_sample=True,
                max_new_tokens=150,
                temperature=temperature,
                num_beams=1,
                top_k=top_k,
                top_p=top_p,
                use_arithmetic=False,
            )
        if "greedy" in sampling_strategies:
            greedy_decoding_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=150
            )
        outputs[arg_key].append({})
        outputs[arg_key][-1]["gt"] = answer
        outputs[arg_key][-1]["question"] = question
        outputs[arg_key][-1]["input_prompt"] = input_prompt
        outputs[arg_key][-1]["metrics"] = {}
        if "arithmetic" in sampling_strategies:
            outputs[arg_key][-1]["arithmetic_sampling"] = [
                sequence.split(COT_TRIGGER)[-1].strip("\n")
                for sequence in tokenizer.batch_decode(
                    arithmetic_sample_outputs, skip_special_tokens=True
                )
            ]
            outputs[arg_key][-1]["metrics"]["arithmetic_ngram_diversity"] = ngram_diversity(
                outputs[arg_key][-1]["arithmetic_sampling"]
            )
            outputs[arg_key][-1]["metrics"]["arithmetic_accuracy"] = numerical_accuracy(
                outputs[arg_key][-1]["arithmetic_sampling"], answer
            )
        if "ancestral" in sampling_strategies:
            outputs[arg_key][-1]["ancestral_sampling"] = [
                sequence.split(COT_TRIGGER)[-1].strip("\n")
                for sequence in tokenizer.batch_decode(
                    ancestral_sample_outputs, skip_special_tokens=True
                )
            ]
            outputs[arg_key][-1]["metrics"]["ancestral_ngram_diversity"] = ngram_diversity(
                outputs[arg_key][-1]["ancestral_sampling"]
            )
            outputs[arg_key][-1]["metrics"]["ancestral_accuracy"] = numerical_accuracy(
                outputs[arg_key][-1]["ancestral_sampling"], answer
            )
        if "greedy" in sampling_strategies:
            outputs[arg_key][-1]["greedy_decoding"] = [
                sequence.split(COT_TRIGGER)[-1].strip("\n")
                for sequence in tokenizer.batch_decode(
                    greedy_decoding_output, skip_special_tokens=True
                )
            ]
            outputs[arg_key][-1]["metrics"]["greedy_ngram_diversity"] = ngram_diversity(
                outputs[arg_key][-1]["greedy_decoding"]
            )
            outputs[arg_key][-1]["metrics"]["greedy_accuracy"] = numerical_accuracy(
                outputs[arg_key][-1]["greedy_decoding"], answer
            )
    with open(output_filename, "w") as output_file:
        json.dump(outputs, output_file)

def analyze_results(output_filename, num_examples, sampling_strategies):
    outputs = {}
    if os.path.exists(output_filename):
        with open(output_filename, "r") as output_file:
            outputs = json.load(output_file)
    for arg_key, data in outputs.items():
        print(f"Results for {arg_key}:")
        arithmetic_ngram_diversity = 0
        ancestral_ngram_diversity = 0
        greedy_ngram_diversity = 0
        arithmetic_accuracy = 0
        ancestral_accuracy = 0
        greedy_accuracy = 0
        for example in data:
            arithmetic_ngram_diversity += example["metrics"]["arithmetic_ngram_diversity"]
            ancestral_ngram_diversity += example["metrics"]["ancestral_ngram_diversity"]
            greedy_ngram_diversity += example["metrics"]["ancestral_ngram_diversity"]
            arithmetic_accuracy += example["metrics"]["arithmetic_accuracy"]
            ancestral_accuracy += example["metrics"]["ancestral_accuracy"]
            greedy_accuracy += example["metrics"]["greedy_accuracy"]
        print(f"Arithmetic N-gram Diversity: {arithmetic_ngram_diversity / num_examples}")
        print(f"Ancestral N-gram Diversity: {ancestral_ngram_diversity / num_examples}")
        print(f"Arithmetic Accuracy: {arithmetic_accuracy / num_examples}")
        print(f"Ancestral Accuracy: {ancestral_accuracy / num_examples}")


if __name__ == "__main__":
    init_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, nargs="*", default=[8])
    parser.add_argument("--model", default="flan-t5-large")
    parser.add_argument("--sampling-strategies", type=str, nargs="*", default=["arithmetic", "ancestral", "greedy"])
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--quant-8-bit", action="store_true")
    parser.add_argument("-D", "--num-examples", type=int, default=30)
    parser.add_argument("-N", "--num-samples", type=int, nargs="*", default=[10])
    parser.add_argument("-T", "--temperature", type=float, nargs="*", default=[1])
    parser.add_argument("-k", "--top-k", type=int, nargs="*", default=[50])
    parser.add_argument("-p", "--top-p", type=float, nargs="*", default=[1.0])
    parser.add_argument("--seed", type=int, default=1379)
    parser.parse_args()
    args = parser.parse_args()

    MODEL = MODEL_ENUM[args.model]
    DATASET = DATASET_ENUM[args.dataset]

    if "flan-t5" in MODEL[1]:
        model_class = AutoModelForSeq2SeqLM
    else:
        model_class = AutoModelForCausalLM
    # model = model_class.from_pretrained(
    #     "/".join(MODEL), load_in_8bit=args.quant_8_bit
    # ).to(DEVICE)
    model = model_class.from_pretrained(
        "/".join(MODEL), load_in_8bit=args.quant_8_bit
    )
    tokenizer = AutoTokenizer.from_pretrained("/".join(MODEL), use_fast=True)

    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id

    dataset = (
        load_HF_dataset(DATASET[0], DATASET[1], DATASET[2])
        .with_format("torch")
        .shuffle(seed=args.seed)
        .select(range(args.num_examples))
    )
    output_filename = f"{args.model}__{args.dataset}__output.json"
    for num_shot in args.num_shots:
        prompt_prefix = generate_few_shot_exemplars(DATASET[0], num_examples=num_shot)
        for N in args.num_samples:
            for T in args.temperature:
                for k in args.top_k:
                    for p in args.top_p:
                        cot_reasoning(
                            output_filename,
                            model,
                            tokenizer,
                            dataset,
                            prompt_prefix,
                            sampling_strategies=args.sampling_strategies,
                            num_samples=N,
                            temperature=T,
                            num_examples=args.num_examples,
                            num_shots=num_shot,
                            top_k=k,
                            top_p=p
                        )
    analyze_results(output_filename=output_filename, num_examples=args.num_examples, sampling_strategies=args.sampling_strategies)
    term_time = time.time()
    print(f"Time taken: {(term_time - init_time):.4f}s")