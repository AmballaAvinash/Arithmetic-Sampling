import os
import json
import argparse
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
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
        input_ids = tokenizer(
            input_prompt, truncation=True, return_tensors="pt"
        ).input_ids.to(DEVICE)
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
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = model_class.from_pretrained(
        "/".join(MODEL), 
        quantization_config = quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("/".join(MODEL), use_fast=True)

    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    input_prompt = """ Summarize the following text in less than 30 words:
    "AMD to open source Micro Engine Scheduler firmware for Radeon GPUs
    11 comment bubble on white
    And it was all thanks to peer pressure
    iconMatthew Connatser
    Fri 5 Apr 2024 // 17:00 UTC
    AMD plans to document and open source its Micro Engine Scheduler (MES) firmware for GPUs, giving users more control over Radeon graphics cards.

    It's part of a larger effort AMD confirmed earlier this week about making its GPUs more open source at both a software level in respect to the ROCm stack for GPU programming and a hardware level. Details were scarce with this initial announcement, and the only concrete thing it introduced was a GitHub tracker.

    However, yesterday AMD divulged more details, specifying that one of the things it would be making open source was the MES firmware for Radeon GPUs. AMD says it will be publishing documentation for MES around the end of May, and will then release the source code some time afterward.

    For one George Hotz and his startup, Tiny Corp, this is great news. Throughout March, Hotz had agitated for AMD to make MES open source in order to fix issues he was experiencing with his RX 7900 XTX-powered AI server box. He had talked several times to AMD representatives, and even the company's CEO, Lisa Su.

    Standardization could open door to third-party chiplets in AMD designs
    Kaby Lake-G chip back from the grave, now on modest firewall-router-NAS mobo
    TSMC's 3nm node powers up, setting stage for tech giants' next-gen chips
    Tiny Corp launches Nvidia-powered AI computer because 'it just works'
    While Hotz was initially "70 percent confident" that MES would be made open source, after his conversation with Su in late March he figured it wasn't going to happen after all. This would have killed the AMD-powered TinyBox, which uses six RX 7900 XTXs, but the project was saved thanks to the discovery of a single "umr" repository that made the box work well enough.

    Still, Tiny Corp had reluctantly decided to launch an RTX 4090-powered TinyBox anyway since it offered a solution that "just works." An Intel version might also be in the pipeline, though the performance of Intel's best GPU, the Arc A770, would be a significant concern since it comes nowhere near either the RX 7900 XTX and RTX 4090.

    Yet Hotz's 70 percent confidence level was right in the end as AMD is seemingly doing exactly what he was asking for. Though making the MES firmware open source is a strange conclusion to the saga, and it's unclear why AMD seems to have changed course now.

    "Spoke with them for months and months and got nothing, despite the public engagement you saw. I think it was treated as a PR problem, not a GPU problem," Hotz speculated on X before AMD announced its open source efforts.

    Hotz has not yet responded to AMD's announcement, and we've asked him and AMD to comment." """
    input_ids = tokenizer(
        input_prompt, truncation=True, return_tensors="pt"
    ).input_ids.to(DEVICE)
    
    model_output_tensor = model.generate(
        input_ids=input_ids,
        # num_return_sequences=num_samples,
        # do_sample=True,
        max_new_tokens=150,
        # temperature=temperature,
        # num_beams=1,
        # top_k=50 if top_k is None else top_k,
        # top_p=1 if top_p is None else top_p, 
        # use_arithmetic=True,
    )
    
    model_output = tokenizer.batch_decode(model_output_tensor, skip_special_tokens=True)
    print(model_output)
