import copy
import random
from typing import Union, List
import logging

import torch
from transformers import StoppingCriteria, LogitsProcessor, LogitsProcessorList
# import openai

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
default_decoding_args = {
    "max_new_tokens": 100,
    "do_sample": False,  # enable sampling
    "top_p": 0,  # nucleus sampling
    "temperature": 0,  # lower makes the distribution sharper
    "min_length": None,
    "use_cache": True,
    "top_k": 100,  # restrict to top-k probability tokens
    "repetition_penalty": 1.,  # 1 means no penalty; up to inf
    "length_penalty": 1.,  # length_penalty > 0.0 == longer sequences; length_penalty < 0.0 == shorter sequences
    "num_beams": 10,  # beam search
    "num_return_sequences": 10,  # number of beams to return
    "no_repeat_ngram_size": 3,
    "renormalize_logits": True,
}
default_metrics = [
    {"name": "rouge", 'score_keys': ['rouge1', 'rouge2', 'rougeL'], 'args': {}},
    {"name": "bleu", 'score_keys': ['bleu'], 'args': {'max_order': 2}},
    {"name": "bertscore", 'score_keys': ['f1'], 'args': {'model_type': 'distilbert-base-uncased'}},
    {"name": "accuracy", 'score_keys': ['accuracy'], 'args': {}},
]


def construct_prompt_from_args(input, input_prefix):
    prompt_arr = []  # this is later converted into a string using "{sep}".join(), where `sep` may be "\n\n"
    
    if input_prefix is None:
        input_prefix = copy.copy(default_input_prefix)
    input_text = f"{input_prefix}{input}"
    prompt_arr.append(input_text)
    return prompt_arr
def prompt_arr_2_text(prompt_arr, prompt_sep, output_prefix):
    # Output prefix (final prompt text)
    if output_prefix is None:
        output_prefix = copy.copy(default_output_prefix)
    else:
        if output_prefix != "":
            prompt_arr.append(output_prefix)
        # if is_beluga and is_chat:
        #     prompt = f"{beluga_DEFAULT_SYSTEM_PROMPT}### User:\n{(prompt_sep.join(prompt_arr))}"
    prompt = prompt_sep.join(prompt_arr)
    return prompt

def construct_args_from_example(d,task_name):
    # breakpoint()
    if 'arc' in task_name:
        question = d['question']
        labels = d['choices']['label']
        options = d['choices']['text']
        answer = d['choices']['text'][d['choices']['label'].index(d['answerKey'])]
        return question,labels,options,answer