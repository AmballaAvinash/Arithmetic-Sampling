import json
import numpy as np
import os
import random
import torch
import datasets
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
from transformers import BitsAndBytesConfig
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, GemmaForCausalLM, LogitsProcessorList, GPT2Tokenizer,GPT2LMHeadModel,T5ForConditionalGeneration, T5Tokenizer


os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_hf_data_set(split,dataset_name, dataset_subname):
        data = {}
        data[split] = datasets.load_dataset(dataset_name,dataset_subname, split="validation",trust_remote_code=True)
        return data[split]

def run_experiments(model_name, data, batch, output_dir, shot, load_in_8bit=False):
    
    # model = "google/flan-t5-large"
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model).to('cuda')
   
    # # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    # model.config.pad_token_id = model.config.eos_token_id
    # model.generation_config.pad_token_id = model.config.eos_token_id
        
        
    # model = ["google/flan-t5-large" ,"meta-llama/Llama-2-7b-hf",  "google/gemma-2b-it", "google/gemma-7b-it"]

    cls_tokenizer = AutoTokenizer
    cls_model = AutoModelForCausalLM
    tokenizer_args = {}
    device_map="auto"
    torch_dtype=torch.float16
    
    if model_name=="llama-2":
        model = "meta-llama/Llama-2-7b-hf"
        cls_model = LlamaForCausalLM 
        cls_tokenizer = LlamaTokenizer
        tokenizer_args.update({"add_bos_token": True, "add_eos_token": False})
        
        # cls_tokenizer.pad_token = cls_tokenizer.eos_token
        # cls_model.config.pad_token_id = cls_model.config.eos_token_id
    elif model_name=="flan-t5":
        model = "google/flan-t5-large"
        cls_model = T5ForConditionalGeneration
        cls_tokenizer = T5Tokenizer
        # torch_dtype = torch.float32  # because of a logsoftmax error with half precision; TODO: double-check
    elif model_name=="gemma":
        model = "google/gemma-2b-it"
        cls_model = AutoModelForCausalLM
        cls_tokenizer = AutoTokenizer
        
    tokenizer = cls_tokenizer.from_pretrained(model, **tokenizer_args)

    if load_in_8bit:
        # breakpoint()
        bnb_config= BitsAndBytesConfig(load_in_8bit=True,)
        model = cls_model.from_pretrained(model,
                                            torch_dtype=torch.bfloat16,
                                            device_map=device_map,
                                            quantization_config=bnb_config,
                                            # low_cpu_mem_usage=low_cpu_mem_usage,
                                            cache_dir = '/work/pi_dhruveshpate_umass_edu/aamballa_umass_edu/models/.cache',
                                            trust_remote_code=True,
                                            )
    
    else:
        model = cls_model.from_pretrained(model,
                                            torch_dtype=torch_dtype,
                                            device_map=device_map,
                                            # low_cpu_mem_usage=low_cpu_mem_usage,
                                            cache_dir = '/work/pi_dhruveshpate_umass_edu/aamballa_umass_edu/models/.cache',
                                            trust_remote_code=True,
                                            load_in_8bit=load_in_8bit)
    
    tokenizer.pad_token =  tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    
        
    
    if shot=="oneshot": 
        default_fwd_instruction = "The German sentence: \"Hier ist die Vita von etwa acht Dutzend Talyschen aus der Gegend dokumentiert, die älter als 100. Fatulayeva zeigt auf ein Schwarz-Weiß-Foto.\" translates to an English sentence: \"Here the lives of eight dozen Talysh from the area who lived to older than 100 are documented. Fatulayeva points out a black & white photo.\". Now, Translate the following German sentence to an English sentence."
    else:
        default_fwd_instruction = "Translate the following German sentence to an English sentence."
    default_fwd_input_prefix = "German sentence: "
    default_fwd_target_prefix = "English sentence: "
    
    results = {}
    for N in [5]: # num_decodes
        for temp in [0.01, 0.5, 1.0]: # temperature
            for p in [0.8, 1.0]: # top p 
                for k in [30, 50]: # top k 
                   output_dict = test(default_fwd_instruction, default_fwd_input_prefix, default_fwd_target_prefix, model_name, model, tokenizer, data, batch, output_dir, shot, N, temp, p, k )
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
                   results["Num Samples"] = len(data)
                   results["Num decodes"] = N
                   results["temperature"] = temp
                   results["Top p "] = p
                   results["Top k"] = k
                   
                   results["BLEU Arithmetic"] = avg_bleu_arith_toppk
                   results["BLEU Sampling"] = avg_bleu_temp_toppk
                   results["N gram diversity Arithmetic"] = avg_ngram_arith_toppk
                   results["N gram diversity Sampling"] = avg_ngram_temp_toppk
                   
                   with open(f'{output_dir}/{model_name}__{shot}__Samples_{len(data)}__N_{N}__temp_{temp}__p_{p}__k_{k}_results.json','w') as f:
                       json.dump(results,f)
                   
                   
                   

def test(default_fwd_instruction, default_fwd_input_prefix, default_fwd_target_prefix,  model_name, model, tokenizer, data, batch, output_dir, shot, N = 1, temp = 1, p=1, k = 50):
    output_dict = {}
    
    for idx, d in enumerate(tqdm(data, desc="Predicting")):
        if idx%batch==0:
            input_batch =[]
            
        prompt_arr = [default_fwd_instruction,default_fwd_input_prefix]
        prompt_arr.append(d['de'])
        prompt_arr.append(default_fwd_target_prefix)
        input_prompt = (' ').join(prompt_arr)  # join the sentences
        input_batch.append(input_prompt)
        
        output_dict[idx] = {}
        output_dict[idx]['gt'] = d['en']
        
        if (idx+1)%batch==0:
            input_ids = tokenizer(input_batch, truncation=True, padding = True, return_tensors="pt").input_ids.to('cuda')

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
            
            
            
            decode_arith_toppk = [i.split('English sentence: ')[-1].strip('\n').replace("<pad>", "") for i in tokenizer.batch_decode(outputs_arith_toppk, skip_special_tokens=True)]  
            decode_temp_toppk = [i.split('English sentence: ')[-1].strip('\n').replace("<pad>", "") for i in tokenizer.batch_decode(outputs_temp_toppk, skip_special_tokens=True)]   
                   
            num_decodes = 0
            for j in range(idx-batch+1,idx+1):
                output_dict[j]['arith_toppk'] = decode_arith_toppk[num_decodes:num_decodes+N]
                output_dict[j]['temp_toppk'] = decode_temp_toppk[num_decodes:num_decodes+N]
               
                output_dict[j]['bleu_score_arith_toppk'], output_dict[j]['n_gram_div_arith_toppk'] = calculate_bleu_and_ngram_diversity(output_dict[j]['gt'], output_dict[j]['arith_toppk'])
                output_dict[j]['bleu_score_temp_toppk'], output_dict[j]['n_gram_div_temp_toppk'] = calculate_bleu_and_ngram_diversity(output_dict[j]['gt'], output_dict[j]['temp_toppk'])
           
                num_decodes+=N
        
     
    with open(f'{output_dir}/{model_name}__{shot}__Samples_{len(data)}__N_{N}__temp_{temp}__p_{p}__k_{k}_output.json','w') as f:
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
    samplesize = 100 
    batch = 2
    output_dir = "outputs/WMT/V1"
    
    random.seed(41)
    data =  random.sample(load_hf_data_set('validation','wmt14','de-en')['translation'],samplesize)

    
    # it versions and quantized
    for model_name in ["gemma"]:  #"llama-2", "flan-t5"
        for shot in ["zeroshot", "oneshot"]:
            print(f"{model_name}, {shot}")
            run_experiments(model_name, data, batch, output_dir, shot, load_in_8bit=False)
    
 