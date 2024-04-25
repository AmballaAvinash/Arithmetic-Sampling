#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 00:07:07 2024

@author: avinashamballa
"""
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, GemmaForCausalLM, LogitsProcessorList, GPT2Tokenizer,GPT2LMHeadModel,T5ForConditionalGeneration, T5Tokenizer, MT5ForConditionalGeneration, M2M100ForConditionalGeneration, M2M100Tokenizer
from mbr import MBR, MBRConfig
from transformers import pipeline


model = "meta-llama/Llama-2-7b-hf"
cls_model = LlamaForCausalLM
cls_tokenizer = LlamaTokenizer

tokenizer = cls_tokenizer.from_pretrained(model)
model = MBR(cls_model).from_pretrained(model)

mbr_config = MBRConfig(
    num_samples=10,
    metric="chrf",
)

input_batch = ["Hi how are you?", "I took my cat for a walk."]
input_ids = tokenizer(input_batch, truncation=True, padding = True, return_tensors="pt").input_ids



outputs_temp_toppk = model.generate(
    input_ids = input_ids,
    num_return_sequences = 1,
    do_sample = True,
    temperature = 0.5,
    top_p=0.8,
    top_k=50,
    num_beams = 1,
    max_new_tokens = 100,
    mbr_config=mbr_config,
    tokenizer=tokenizer
    )


print(outputs_temp_toppk)


generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator("Hello,", mbr_config=mbr_config, tokenizer=tokenizer)
print(output)