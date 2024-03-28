from collections import Counter
import copy
import random
from typing import Union, List
import logging
import re
import torch
from transformers import StoppingCriteria, LogitsProcessor, LogitsProcessorList
# import openai

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# default_strat_qa_instruction = "Like each of the previous examples, answer the following question with Yes or No, and provide the reasoning as demonstrated earlier. The reasoning should end with the sentence(with <your_answer> substituted by your final answer):  So the answer is <your_answer>. The answer should be either Yes or No."
default_strat_qa_instruction = """Like each of the previous examples, answer the question with either "Yes" or "No".
Provide reasoning for your answer.
End your reasoning with the sentence: "So the answer is <your_answer>". Replace "<your_answer>" with your final answer, which should be either "Yes" or "No"."""
default_input_prefix = "Input: "
default_output_prefix = "Output: "
default_answer_prefix = "Answer: "
default_question_prefix = "Question: "
default_reasoning_prefix = "Reasoning: "
default_decoding_args = {
    "max_new_tokens": 200,
    "do_sample": False,  # enable sampling
    "top_p": 1.,  # nucleus sampling
    "temperature": 1.,  # lower makes the distribution sharper
    "top_k" : 50,
    "min_length": None,
    # "use_cache": True,
    # "top_k": 100,  # restrict to top-k probability tokens
    # "repetition_penalty": 1.,  # 1 means no penalty; up to inf
    # "length_penalty": 1.,  # length_penalty > 0.0 == longer sequences; length_penalty < 0.0 == shorter sequences
    "num_beams": 1,  # beam search
    "num_return_sequences": 1,  # number of beams to return
    # "no_repeat_ngram_size": 3,
    "renormalize_logits": True,
}
default_metrics = [
    {"name": "rouge", 'score_keys': ['rouge1', 'rouge2', 'rougeL'], 'args': {}},
    {"name": "bleu", 'score_keys': ['bleu'], 'args': {'max_order': 2}},
    {"name": "bertscore", 'score_keys': ['f1'], 'args': {'model_type': 'distilbert-base-uncased'}},
    {"name": "accuracy", 'score_keys': ['accuracy'], 'args': {}},
]
task_logits_processors = {}
class TruncateLogitsProcessor(LogitsProcessor):
    def __init__(self,token_id: Union[int, List[int]],stop_word:Union[str, List[str]],eos_token_id: Union[int, List[int]],tokenizer):
        if isinstance(token_id, int):
            token_id = [token_id]
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if not all(isinstance(i,int) for i in token_id) or any(i < 0 for i in token_id):
            logger.warning(f"`token_id` has to be a list of positive integers, but is {token_id}")
        self.token_id = token_id
        self.stop_word = stop_word
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer
    def __call__(self,input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        max_score_ids = torch.argmax(scores, dim=1)
        for i in range(len(max_score_ids)):
            if (max_score_ids[i] in self.tokenizer(self.stop_word).input_ids) :
                scores[i][:] = -float('inf')
                scores[i][self.eos_token_id[0]] = 0
        scores.to(input_ids.device)
        if torch.argmax(scores[:,]) in self.token_id:
            print('yes')
            scores = torch.zeros(scores.shape)
            scores[:, self.eos_token_id] = 1
                
        return scores
def construct_qa_prompt_from_args( question, question_prefix,
                                    reasoning=None, reasoning_prefix=None,
                                     answer=None, answer_prefix=None):
    prompt_arr = []  # this is later converted into a string using "{sep}".join(), where `sep` may be "\n\n"
    # The test or demonstration question
    if question_prefix is None:
        question_prefix = copy.copy(default_question_prefix)
    question_text = f"{question_prefix}{question}"
    prompt_arr.append(question_text)
    if reasoning is not None and reasoning_prefix is not None:
        reasoning_text = f"{reasoning_prefix}{reasoning}"
        if reasoning_text != "":
            prompt_arr.append(reasoning_text)
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
    if 'strategy_qa' in task_name:
        question = d['input']
        answer = [k for k in d['target_scores'].keys() if d['target_scores'][k] == 1][0].lower()
        target = d['target']
        return {'question':question,
                'reasoning':target,
                "answer" : answer,
                'instructions': default_strat_qa_instruction,
                "question_prefix" : default_question_prefix,
                "answer_prefix": default_answer_prefix,
                "reasoning_prefix":default_reasoning_prefix,
                "n_shots" : 6,
                "demos_split":'demo',
                "task_name" : task_name,
                'construct_args_fn' : construct_args_from_example}, answer
    if 'arc' in task_name:
        question = d['question']
        labels = d['choices']['label']
        options = d['choices']['text']
        answer = d['choices']['text'][d['choices']['label'].index(d['answerKey'])]
        return question,labels,options,answer
def fix_posthoc(decoded,task_name):
    if 'strategy_qa' in task_name:
        labels = []
        for d in decoded:
            patterns = {
            "the answer is" : r"the\s+answer\s+is\s+(\w+)",
            "so the answer is": r"so\s+the\s+answer\s+is\s+(\w+)",
            "so the answer to the question is": r"so\s+the\s+answer\s+to\s+the\s+question\s+is\s+(\w+)",
            "answer": r"answer:\s+(\w+)"
            }
            matches = []
            for phrase, pattern in patterns.items():
                d = d.replace(',','').lower()
                match = re.search(pattern, d)
                if match:
                    matches.append(match.group(1))
            # breakpoint()
            try:
                majority_match = max( Counter(matches), key=Counter(matches).get)
                labels.append(majority_match)
            except:
                print(f"Answer couldn't be extracted: {d}")
                labels.append('nan')
        if len([x for x in labels if x != 'nan']) > 0:
            labels = [x for x in labels if x != 'nan']
        majority_label = max( Counter(labels), key=Counter(labels).get)
        # breakpoint()
    return majority_label