import json
import numpy as np
import os
import random
import torch
import datasets
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_hf_data_set(split,dataset_name, dataset_subname):
        data = {}
        data[split] = datasets.load_dataset(dataset_name,dataset_subname, split="validation",trust_remote_code=True)
        return data[split]

def run_experiments(data):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    
    default_fwd_instruction = "Translate the following German sentence to an English sentence."
    default_fwd_input_prefix = "German sentence: "
    default_fwd_target_prefix = "English sentence: "
    prompt_arr = [default_fwd_instruction,default_fwd_input_prefix]
    
    results = {}
    for N in [5, 10]: # num_decodes
        for temp in [0.5, 1.0]: # temperature
            for p in [0.8, 1.0]: # top p 
                for k in [30, 50]: # top k 
                   output_dict = test(prompt_arr, model, tokenizer, data, default_fwd_target_prefix, N, temp, p, k )
                   avg_bleu_arith_toppk = []
                   avg_bleu_temp_toppk = []
                   avg_ngram_arith_toppk = []
                   avg_ngram_temp_toppk = []
                   for idx in range(len(data)):   
                       avg_bleu_arith_toppk.append(output_dict[idx]['bleu_score_arith_toppk'])
                       avg_ngram_arith_toppk.append(output_dict[idx]['n_gram_div_arith_toppk'])
                       avg_bleu_temp_toppk.append(output_dict[idx]['bleu_score_temp_toppk'])
                       avg_ngram_temp_toppk.append(output_dict[idx]['n_gram_div_temp_toppk'])
                       
                   avg_bleu_arith_toppk = np.mean(avg_bleu_arith_toppk)
                   avg_bleu_temp_toppk = np.mean(avg_bleu_temp_toppk)
                   avg_ngram_arith_toppk = np.mean(avg_ngram_arith_toppk)
                   avg_ngram_temp_toppk = np.mean(avg_ngram_temp_toppk)
                   
                   results = {}
                   results["Num decodes"] = N
                   results["temperature"] = temp
                   results["Top p "] = p
                   results["Top k"] = k
                   
                   results["BLEU Arithmetic"] = avg_bleu_arith_toppk
                   results["BLEU Sampling"] = avg_bleu_temp_toppk
                   results["N gram diversity Arithmetic"] = avg_ngram_arith_toppk
                   results["N gram diversity Sampling"] = avg_ngram_temp_toppk
                   
                   with open(f'outputs/flan_t5_wmt14_de-en__N_{N}__temp_{temp}__p_{p}__k_{k}_results.json','w') as f:
                       json.dump(results,f)
                   
                   
                   

def test(prompt_arr, model, tokenizer, data, default_fwd_target_prefix, N = 1, temp = 1, p=1, k = 50):
    output_dict = {}
    for idx, d in enumerate(tqdm(data, desc="Predicting")):
        prompt_arr.append(d['de'])
        prompt_arr.append(default_fwd_target_prefix)
        input_prompt = (' ').join(prompt_arr)  # join the sentences
        input_ids = tokenizer(input_prompt, truncation=True, return_tensors="pt").input_ids
       
        
        outputs_arith_toppk = model.generate(
            input_ids = input_ids,
            num_return_sequences = N,
            do_sample = True,
            temperature = temp,
            top_p=p,
            top_k=k,
            num_beams = 1,
            max_new_tokens = 100,
            use_arithmetic = True
            )
        
        outputs_temp_toppk = model.generate(
            input_ids = input_ids,
            num_return_sequences = N,
            do_sample = True,
            temperature = temp,
            top_p=p,
            top_k=k,
            num_beams = 1,
            max_new_tokens = 100,
            use_arithmetic = False
            )
        
        output_dict[idx] = {}
        output_dict[idx]['gt'] = d['en']
        
        output_dict[idx]['arith_toppk'] = [i.split('English sentence: ')[-1].strip('\n') for i in tokenizer.batch_decode(outputs_arith_toppk, skip_special_tokens=True)]
        output_dict[idx]['temp_toppk'] = [i.split('English sentence: ')[-1].strip('\n') for i in tokenizer.batch_decode(outputs_temp_toppk, skip_special_tokens=True)]
       
        output_dict[idx]['bleu_score_arith_toppk'], output_dict[idx]['n_gram_div_arith_toppk'] = calculate_bleu_and_ngram_diversity(output_dict[idx]['gt'], output_dict[idx]['arith_toppk'])
        output_dict[idx]['bleu_score_temp_toppk'], output_dict[idx]['n_gram_div_temp_toppk'] = calculate_bleu_and_ngram_diversity(output_dict[idx]['gt'], output_dict[idx]['temp_toppk'])
    with open(f'outputs/flan_t5_wmt14_de-en__N_{N}__temp_{temp}__p_{p}__k_{k}_output.json','w') as f:
        json.dump(output_dict,f)
        
    return output_dict

def calculate_bleu_and_ngram_diversity(reference, translations):
    translations_split = [x.split() for x in translations]
    bleu_score = np.mean([ sentence_bleu([reference.split()], x,  smoothing_function=SmoothingFunction().method4) for x in translations_split])

    n_values = [1, 2, 3, 4]  # BLUE-4 and ngram-4
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
        ngram_diversity_score += total_unique_ngrams / (total_ngram_count + torch.finfo(torch.float32).eps)
    return bleu_score, ngram_diversity_score

if __name__ == "__main__":
    samplesize = 500
    data =  random.sample(load_hf_data_set('validation','wmt14','de-en')['translation'],samplesize)
    run_experiments(data)
