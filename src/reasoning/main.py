import json
import os
import random
import torch
import datasets
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from utils import create_demo_text

# Reference: https://github.com/kojima-takeshi188/zero_shot_cot/blob/main/main.py

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

ZERO_SHOT_ANSWER_TRIGGER = "\nTherefore, the answer (numeric) is"
ZERO_SHOT_COT_TRIGGER = "Let's think step by step."
FEW_SHOT_ANSWER_TRIGGER = "The answer (numeric) is"
FEW_SHOT_COT_TRIGGER = "Let's think step by step."

MODEL = ("google","flan-t5-large")
# MODEL = ("google", "gemma-2b-it")
DATASET = ("test", "gsm8k", "main")


def load_hf_data_set(split, dataset_name, dataset_subname):
    data = {}
    return datasets.load_dataset(
        dataset_name, dataset_subname, split=split, trust_remote_code=True
    )


def run_experiments(zero_shot_cot_flag=False):
    tokenizer = AutoTokenizer.from_pretrained("/".join(MODEL), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained("/".join(MODEL), load_in_4bit=True).to("cuda")
    torch.device = "cuda"
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    dataset = load_hf_data_set(*DATASET).with_format("torch")
    data = dataset.shuffle(seed=42).select(range(10))
    demo = create_demo_text(FEW_SHOT_ANSWER_TRIGGER) if not zero_shot_cot_flag else ""
    for N in [5, 10]:
        for temp in [0.5, 1.0]:
            reasoning_test(model, tokenizer, data, demo, N, temp)


def reasoning_test(model, tokenizer, data, demo="", N=10, temp=0.5):
    output_dict = {}
    for idx, d in enumerate(tqdm(data, desc="Predicting")):
        x, y = d["question"], d["answer"]
        x = "Q: " + x[0] + "\n" + "A:"
        y = y.split("####")[-1].strip()
        if demo != "":
            x = x + " " + ZERO_SHOT_COT_TRIGGER
        else:
            x = demo + x
        input_prompt = x
        input_ids = tokenizer(
            input_prompt, truncation=True, return_tensors="pt"
        ).input_ids.to("cuda")
        outputs_arith = model.generate(
            input_ids=input_ids,
            num_return_sequences=N,
            do_sample=True,
            max_new_tokens=100,
            temperature=temp,
            num_beams=1,
            use_arithmetic=True,
        )
        outputs_sample = model.generate(
            input_ids=input_ids,
            num_return_sequences=N,
            do_sample=True,
            temperature=temp,
            num_beams=1,
            max_new_tokens=100,
            use_arithmetic=False,
        )
        output_dict[idx] = {}
        output_dict[idx]["gt"] = y
        output_dict[idx]["arithmetic"] = [
            i.split(ZERO_SHOT_COT_TRIGGER)[-1].strip("\n")
            for i in tokenizer.batch_decode(outputs_arith, skip_special_tokens=True)
        ]
        output_dict[idx]["sampling"] = [
            i.split(ZERO_SHOT_COT_TRIGGER)[-1].strip("\n")
            for i in tokenizer.batch_decode(outputs_sample, skip_special_tokens=True)
        ]
    with open(f"{MODEL[1]}__{DATASET[1]}__N_{N}__temp_{temp}__output.json", "w") as f:
        json.dump(output_dict, f)


def calculate_bleu_and_ngram_diversity(reference, translations):
    bleu_score = sentence_bleu(
        [reference] * len(translations),
        translations,
        smoothing_function=SmoothingFunction().method4,
    )
    n_values = [1, 2, 3, 4]
    total_unique_ngrams = 0
    ngram_diversity_score = 0
    for n in n_values:
        unique_ngrams = set()
        total_ngram_count = 0
        for translation in translations:
            # Compute n-grams
            translation_ngrams = list(ngrams(translation.split(), n))
            # Count unique n-grams
            total_ngram_count += len(list(translation_ngrams))
            unique_ngrams.update(translation_ngrams)
        # Update total counts
        total_unique_ngrams = len(unique_ngrams)
        ngram_diversity_score += total_unique_ngrams / (
            total_ngram_count + torch.finfo(torch.float32).eps
        )
    return bleu_score, ngram_diversity_score


if __name__ == "__main__":
    run_experiments(zero_shot_cot_flag=True)
