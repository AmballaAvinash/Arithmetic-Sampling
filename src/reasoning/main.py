import os
import json
import argparse
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, set_seed
from accelerate import Accelerator
from tqdm import tqdm
from utils import *

# Reference: https://github.com/kojima-takeshi188/zero_shot_cot/blob/main/main.py

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

intermediate_fix = "\n\nNow, answer the following question:"
COT_TRIGGER = "\nA: Let's think step by step."
hard_prefix = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8."""

def cot_reasoning(
    output_filename, model, it_flag, tokenizer, data, prompt_prefix, sampling_strategies, num_samples, temperature, num_examples, num_shots, top_k, top_p
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
        input_prompt = hard_prefix + intermediate_fix + f"\n\nQ: {question}{COT_TRIGGER}"
        # # Uncomment as needful
        # input_prompt = f"Question: {question} \nAnswer: {COT_TRIGGER}"
        # if len(prompt_prefix):
        #     input_prompt = prompt_prefix + input_prompt
        # if it_flag:
        #     chat = [
        #         {"role": "user", "content": input_prompt}
        #     ]
        #     input_prompt = tokenizer.apply_chat_template(chat, tokenize = False, add_generation_prompt=True)
        prompt_tokenized = tokenizer(
            input_prompt, return_tensors="pt", return_token_type_ids=False
        ).to(DEVICE)
        accelerator = Accelerator()
        accelerator.wait_for_everyone()
        if "arithmetic" in sampling_strategies:
            if num_samples > 5:
                new_num_samples = num_samples // accelerator.num_processes
                with accelerator.split_between_processes(range(accelerator.num_processes)) as idx:
                    with torch.no_grad():
                        arithmetic_sample_outputs = model.generate(
                            **prompt_tokenized,
                            num_return_sequences=new_num_samples,
                            do_sample=True,
                            max_new_tokens=256,
                            temperature=temperature,
                            num_beams=1,
                            top_k=50 if top_k is None else top_k,
                            top_p=1 if top_p is None else top_p, 
                            use_arithmetic=True,
                        )
            else:
                with torch.no_grad():
                    arithmetic_sample_outputs = model.generate(
                    **prompt_tokenized,
                    num_return_sequences=num_samples,
                    do_sample=True,
                    max_new_tokens=256,
                    temperature=temperature,
                    num_beams=1,
                    top_k=50 if top_k is None else top_k,
                    top_p=1 if top_p is None else top_p, 
                    use_arithmetic=True,
                )
        if "ancestral" in sampling_strategies:
            if num_samples > 5:
                new_num_samples = num_samples // accelerator.num_processes
                with accelerator.split_between_processes(range(accelerator.num_processes)) as idx:
                    with torch.no_grad():
                        ancestral_sample_outputs = model.generate(
                            **prompt_tokenized,
                            num_return_sequences=new_num_samples,
                            do_sample=True,
                            max_new_tokens=256,
                            temperature=temperature,
                            num_beams=1,
                            top_k=50 if top_k is None else top_k,
                            top_p=1 if top_p is None else top_p, 
                            use_arithmetic=False,
                        )
            else:
                with torch.no_grad():
                    ancestral_sample_outputs = model.generate(
                        **prompt_tokenized,
                        num_return_sequences=num_samples,
                        do_sample=True,
                        max_new_tokens=256,
                        temperature=temperature,
                        num_beams=1,
                        top_k=50 if top_k is None else top_k,
                        top_p=1 if top_p is None else top_p, 
                        use_arithmetic=False,
                    )
        if "greedy" in sampling_strategies:
            greedy_decoding_output = model.generate(
                **prompt_tokenized,
                max_new_tokens=256
            )
        outputs[arg_key].append({})
        outputs[arg_key][-1]["gt"] = answer
        outputs[arg_key][-1]["question"] = question
        outputs[arg_key][-1]["input_prompt"] = input_prompt
        if "arithmetic" in sampling_strategies:
            outputs[arg_key][-1]["arithmetic_sampling"] = [
                sequence.split(COT_TRIGGER)[-1].split("\n\nQ")[0].strip()
                for sequence in tokenizer.batch_decode(
                    arithmetic_sample_outputs, skip_special_tokens=True
                )
            ]
        if "ancestral" in sampling_strategies:
            outputs[arg_key][-1]["ancestral_sampling"] = [
                sequence.split(COT_TRIGGER)[-1].split("\n\nQ")[0].strip()
                for sequence in tokenizer.batch_decode(
                    ancestral_sample_outputs, skip_special_tokens=True
                )
            ]
        if "greedy" in sampling_strategies:
            outputs[arg_key][-1]["greedy_decoding"] = [
                sequence.split(COT_TRIGGER)[-1].split("\n\nQ")[0].strip()
                for sequence in tokenizer.batch_decode(
                    greedy_decoding_output, skip_special_tokens=True
                )
            ]
        with open(output_filename+"_temp.txt", "a") as output_file:
            json.dump(outputs[arg_key][-1], output_file)    
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
    parser.add_argument("-O", "--num-examples-offset", type=int, default=0)
    parser.add_argument("-N", "--num-samples", type=int, nargs="*", default=[10])
    parser.add_argument("-T", "--temperature", type=float, nargs="*", default=[1])
    parser.add_argument("-k", "--top-k", type=int, nargs="*", default=[50])
    parser.add_argument("-p", "--top-p", type=float, nargs="*", default=[1.0])
    parser.add_argument("--seed", type=int, nargs="*", default=[1379])
    parser.add_argument("--output-path", type=str, default="")
    parser.parse_args()
    args = parser.parse_args()

    MODEL = MODEL_ENUM[args.model]
    DATASET = DATASET_ENUM[args.dataset]

    if "flan-t5" in MODEL[1]:
        model_class = AutoModelForSeq2SeqLM
    else:
        model_class = AutoModelForCausalLM
    if args.quant_8_bit:
        model = model_class.from_pretrained(
            "/".join(MODEL), 
            quantization_config = BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto"
        )
    else:
        model = model_class.from_pretrained(
            "/".join(MODEL)
        ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("/".join(MODEL), use_fast=True)
    print("Tokenizer max input length:", tokenizer.model_max_length)

    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id

    for seed in args.seed:
        dataset = (
            load_HF_dataset(DATASET[0], DATASET[1], DATASET[2])
            .with_format("torch")
            .shuffle(seed=seed)
        )
        dataset = dataset.select(
            range(
                max(0, args.num_examples_offset), 
                min(args.num_examples_offset + args.num_examples, dataset.num_rows)
            )
        )
        set_seed(seed)
        output_filename = args.output_path if len(args.output_path) != 0 else f"{args.model}__{args.dataset}__{seed}_output.json"
        for num_shot in args.num_shots:
            # prompt_prefix = generate_few_shot_exemplars(DATASET[0], num_examples=num_shot)
            for N in args.num_samples:
                for T in args.temperature:
                    for k in args.top_k:
                        for p in args.top_p:
                            cot_reasoning(
                                output_filename,
                                model,
                                "-it" in MODEL[1],
                                tokenizer,
                                dataset,
                                prompt_prefix = None,
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
