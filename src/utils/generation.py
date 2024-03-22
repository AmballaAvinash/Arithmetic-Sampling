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
default_input_prefix = "Input: "
default_output_prefix = "Output: "
default_answer_prefix = "Answer: "
default_question_prefix = "Question: "
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
class TruncateLogitsProcessor(LogitsProcessor):
    def __init__(self,token_id: Union[int, List[int]],eos_token_id: Union[int, List[int]],tokenizer):
        if isinstance(token_id, int):
            token_id = [token_id]
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if not all(isinstance(i,int) for i in token_id) or any(i < 0 for i in token_id):
            logger.warning(f"`token_id` has to be a list of positive integers, but is {token_id}")
        self.token_id = token_id
        
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer
    def __call__(self,input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        max_score_ids = torch.argmax(scores, dim=1)
        for i in range(len(max_score_ids)):
            if (max_score_ids[i] in self.token_id) or (self.tokenizer.decode(max_score_ids[i]).find(':') != -1) :
                scores[i][:] = -float('inf')
                scores[i][self.eos_token_id[0]] = 0
        scores.to(input_ids.device)
        if torch.argmax(scores[:,]) in self.token_id:
            print('yes')
            scores = torch.zeros(scores.shape)
            scores[:, self.eos_token_id] = 1
                
        return scores
def construct_qa_prompt_from_args( answer, answer_prefix,
                                     question=None, question_prefix=None):
    prompt_arr = []  # this is later converted into a string using "{sep}".join(), where `sep` may be "\n\n"
    # The test or demonstration question
    if question_prefix is None:
        question_prefix = copy.copy(default_question_prefix)
    question_text = f"{question_prefix}{question}"
    prompt_arr.append(question_text)
    if answer is not None and answer_prefix is not None:
        answer_text = f"{answer_prefix}{answer}"
        if answer_text != "":
            prompt_arr.append(answer_text)
    return prompt_arr
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
    prompt = prompt_sep.join(prompt_arr)
    return prompt

def construct_args_from_example(d,task_name):
    breakpoint()
    if 'strategy_qa' in task_name:
        question = d['input']
        answer = [k for k in d['target_scores'].keys() if d['target_scores'][k] == 1][0].lower()
        target = d['target']
        return {'question':question,'answer':answer},target
    if 'arc' in task_name:
        question = d['question']
        labels = d['choices']['label']
        options = d['choices']['text']
        answer = d['choices']['text'][d['choices']['label'].index(d['answerKey'])]
        return question,labels,options,answer
def fix_posthoc(decoded,task_name):
    return decoded