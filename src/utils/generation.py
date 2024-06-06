from collections import Counter
import copy
import numpy as np
import random
from typing import Union, List, Dict
import logging
import re
import torch
from transformers import StoppingCriteria, LogitsProcessor, LogitsProcessorList

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

default_strat_qa_instruction = """Like each of the previous examples, answer the question with a reasoning and a final answer: either "Yes" or "No".
Think step by step and provide a reasoning first, which leads to the final answer. Do not start with the final answer. The reasoning should not be too long and should end with the final answer (Yes/No) like the examples.The question has enough information to answer it strategically. Use logical reasoning and available knowledge to respond with either 'Yes' or 'No.'
End your reasoning with the sentence: "So the answer is <your_answer>. Answer: <your_answer>". Replace "<your_answer>" with your final answer, which should be either "Yes" or "No"."""
default_commonsense_qa_instruction = """Like each of the previous examples,answer the following question by thinking logically, from the information provided. The answer should be in a similar format as the examples.
The question has enough information for you to answer, use commonsense reasoning to answer the question to the best of your ability with an answer label(a,b,c,d,e,etc.) by logical deduction and elimination of incorrect answers.
You should end your answer with : "So the answer is <your_answer_choice>". Replace "<your_answer_choice>" with your final answer, which should be one of the answer choice label letters (a,b,c,d,e,etc.)."""
default_gsm8k_instruction = """Like each of the previous examples, answer the following question by thinking logically, from the information provided. The answer should be in a similar format as the examples.
The question has enough information for you to answer, use arithmetic reasoning to answer the question to the best of your ability with a numerical answer by logical deduction and chain-of-thought reasoning.
You should end your answer with : "So the answer is <your_answer>". Replace "<your_answer>" with your final answer, which should be a numerical value."""
default_mt_fwd_instruction = "Translate the following {source} sentence to {target}:"
default_mt_fwd_input_prefix = "{source}: "
default_mt_fwd_target_prefix = "{target}: "
default_input_prefix = "Input: "
default_output_prefix = "Output: "
default_answer_prefix = "Answer: "
default_question_prefix = "Question: "
default_reasoning_prefix = "Reasoning: "
default_decoding_args = {
    "max_new_tokens": 300,
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
            if (max_score_ids[i] in self.tokenizer(self.stop_word).input_ids or self.tokenizer.decode(max_score_ids[i]).find(self.stop_word) != -1) :
                scores[i][:] = -float('inf')
                scores[i][self.eos_token_id[0]] = 0
        scores.to(input_ids.device)
        if torch.argmax(scores[:,]) in self.tokenizer(self.stop_word).input_ids:
            print('yes')
            scores = torch.zeros(scores.shape)
            scores[:, self.eos_token_id] = 1
        return scores

def get_logprob_score(target, prefix, model, tokenizer, len_norm=True, bsz=10):
    bos_added = eos_added = False
    try:
        bos_added = tokenizer.add_bos_token
    except:
        pass
    try:
        eos_added = tokenizer.add_eos_token
    except:
        pass
    scores = []

    if 't5' in tokenizer.name_or_path:
        prefix_tkn = tokenizer(prefix, return_tensors="pt", padding=True)
        prefix_input_ids = prefix_tkn.input_ids
        target_tkn = tokenizer(target, return_tensors="pt", padding=True)
        target_input_ids = target_tkn.input_ids
        target_lens = target_tkn.attention_mask.sum(dim=1)

        for batch_i in range(0, len(prefix_input_ids), bsz):
            batch_prefix_input_ids = prefix_input_ids[batch_i:batch_i + bsz]
            batch_prefix_att_mask = prefix_tkn.attention_mask[batch_i:batch_i + bsz]
            batch_target_input_ids = target_input_ids[batch_i:batch_i + bsz]
            batch_target_lens = target_lens[batch_i:batch_i + bsz]
            with torch.no_grad():
                forward = model(input_ids=batch_prefix_input_ids.cuda(),
                                attention_mask=batch_prefix_att_mask.cuda(),
                                labels=batch_target_input_ids.cuda())
            for i, logits in enumerate(forward.logits):
                target_logits = logits[:batch_target_lens[i]]
                target_logprobs_all = torch.log_softmax(target_logits, dim=-1, dtype=torch.float)
                target_logprobs_seq = target_logprobs_all[
                    torch.arange(batch_target_lens[i]), batch_target_input_ids[i][:batch_target_lens[i]]]
                target_logprob_score = target_logprobs_seq.mean() if len_norm else target_logprobs_seq.sum()
                scores.append(target_logprob_score.item())
    else:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        prefix_input_ids = tokenizer(prefix).input_ids
        target_input_ids = tokenizer(target).input_ids
        if eos_added:
            # Remove EOS from prefix and suffix
            prefix_input_ids = list(map(lambda x: x[:-1], prefix_input_ids))
            target_input_ids = list(map(lambda x: x[:-1], target_input_ids))
        if bos_added:
            # Remove BOS from suffix
            target_input_ids = list(map(lambda x: x[1:], target_input_ids))
        full_input_ids = list(map(lambda x: x[0] + x[1], zip(prefix_input_ids, target_input_ids)))
        # Add padding
        max_len = max(map(len, full_input_ids))
        full_input_ids = list(map(lambda x: x + [pad_id] * (max_len - len(x)), full_input_ids))

        for batch_i in range(0, len(full_input_ids), bsz):
            batch_input_ids = full_input_ids[batch_i:batch_i + bsz]
            with torch.no_grad():
                forward = model(
                    torch.tensor(
                        batch_input_ids).cuda())  # , attention_mask=torch.tensor(batch_attention_mask).cuda()
            for i, logits in enumerate(forward.logits):
                cur_i = batch_i + i
                target_logits = logits[
                                len(prefix_input_ids[
                                        cur_i]) - 1:-1]  # shifted since calcs. are for the next token position
                target_logprobs_all = torch.log_softmax(target_logits, dim=-1, dtype=torch.float)
                target_logprobs_seq = target_logprobs_all[
                    torch.arange(len(target_input_ids[cur_i])), target_input_ids[cur_i]]
                print(len)
                target_logprob_score = target_logprobs_seq.mean() if len_norm else target_logprobs_seq.sum()
                scores.append(target_logprob_score.item())
    return scores

def construct_qa_prompt_from_args(question: str, question_prefix: str, task_name: str, reasoning: str = None, reasoning_prefix: str = None, answer: str = None, answer_prefix: str = None):
    prompt_arr = []  # this is later converted into a string using "{sep}".join(), where `sep` may be "\n\n"
    # The test or demonstration question
    if 'commonsense_qa' not in task_name:
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
    else:
        if question_prefix is None:
            question_prefix = copy.copy(default_question_prefix)
        question_text = f"{question_prefix}{question}"
        prompt_arr.append(question_text)
        if answer is not None and answer_prefix is not None:
            answer_text = f"{answer_prefix}{reasoning}"
            if answer_text != "":
                prompt_arr.append(answer_text)
    return prompt_arr

def construct_prompt_from_args(input: str, input_prefix: str):
    prompt_arr = []  # this is later converted into a string using "{sep}".join(), where `sep` may be "\n\n"
    if input_prefix is None:
        input_prefix = copy.copy(default_input_prefix)
    input_text = f"{input_prefix}{input}"
    prompt_arr.append(input_text)
    return prompt_arr

def prompt_arr_2_text(prompt_arr: List[str], prompt_sep: str, output_prefix: str):
    # Output prefix (final prompt text)
    if output_prefix is None:
        output_prefix = copy.copy(default_output_prefix)
    else:
        if output_prefix != "":
            prompt_arr.append(output_prefix)
    prompt = prompt_sep.join(prompt_arr)
    return prompt

def prompt_arr_2_text_mt(prompt_arr, prompt_sep, is_llama2, is_chat, output_prefix):
    # Output prefix (final prompt text)
    if output_prefix is None:
        output_prefix = copy.copy(default_output_prefix)
    else:
        if output_prefix != "":
            prompt_arr.append(output_prefix)
    prompt = prompt_sep.join(prompt_arr)
    return prompt

def construct_args_from_example(d: Dict, task_name):
    # breakpoint()
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
                "option_texts" : ["Yes","No"],
                "options" : ["Yes","No"],
                "demos_split":'demo',
                "task_name" : task_name,
                'construct_args_fn' : construct_args_from_example}, answer
    elif 'commonsense_qa' in task_name:
        options = ['('+ l.lower() + ') ' + t  for l,t in zip(d['choices']['label'],d['choices']['text'])]
        options_text = ' Answer choices: ' +' '.join(options)
        question = d['question'] + options_text
        answer = d['answerKey'].lower()
        reasoning = d['reasoning'] if 'reasoning' in d.keys() else ''
        return {'question':question,
                "answer" : answer,
                'instructions': default_commonsense_qa_instruction,
                "question_prefix" : default_question_prefix,
                "answer_prefix": default_answer_prefix,
                "reasoning_prefix":default_reasoning_prefix,
                "n_shots" : 6,
                "options"  : [o.lower() for o in d['choices']['label']],
                "option_texts" : d['choices']['text'],
                "reasoning" : reasoning,
                "demos_split":'demo',
                "task_name" : task_name,
                'construct_args_fn' : construct_args_from_example}, answer
    elif 'gsm8k' in task_name:
        question = d['question']
        answer = d['answer']
        reasoning = d['reasoning'] if 'reasoning' in d.keys() else ''
        return {'question':question,
                "answer" : answer,
                'instructions': default_gsm8k_instruction,
                "question_prefix" : default_question_prefix,
                "answer_prefix": default_answer_prefix,
                "reasoning_prefix":default_reasoning_prefix,
                "n_shots" : 8,
                "options" : None,
                "option_texts" : None,
                "reasoning" : reasoning,
                "demos_split":'demo',
                "task_name" : task_name,
                'construct_args_fn' : construct_args_from_example}, answer
    elif 'arc' in task_name:
        question = d['question']
        labels = d['choices']['label']
        options = d['choices']['text']
        answer = d['choices']['text'][d['choices']['label'].index(d['answerKey'])]
        return question,labels,options,answer
    elif 'wmt14' in task_name:
        inf_args = {'input':None,'input_prefix':default_mt_fwd_input_prefix,'output_prefix':default_mt_fwd_target_prefix,'instructions':default_mt_fwd_instruction,'task':task_name}
        # breakpoint()
        return inf_args,None

# def fix_posthoc_mt(decoded,task_name):
#     if 'wmt14' in task_name:
#         # checkp/oint = "bleurt/test_checkpoint"
#         scorer = score.BleurtScorer()
#         utilities = {d:0 for d in decoded}
#         for d in decoded:
#             utilities[d] = sum([scorer.score(references=[d_ref], candidates=[d])[0] for d_ref in decoded])/len(decoded)
#     # breakpoint()

#     # # Calculate BLEURT scores for each decoded sentence against all references
#     # bleurt_scores = np.array([scorer.score(references=decoded, candidates=[d])[0] for d in decoded])

#     # # Compute utilities
#     # utilities2 = np.mean(bleurt_scores)
#     # breakpoint()
#     return utilities


def fix_posthoc(decoded,task_name):
    if 'strategy_qa' in task_name:
        labels = []
        maj_match = {"no":"no","not":"no","n":"no"}
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
                if majority_match in maj_match.keys():
                    labels.append(maj_match[majority_match])
                else:
                    labels.append(majority_match)
            except:
                print(f"Answer couldn't be extracted: {d}")
                labels.append('no')
        if len([x for x in labels if x != 'nan']) > 0:
            labels = [x for x in labels if x != 'nan']
        majority_label = max( Counter(labels), key=Counter(labels).get)
    elif 'commonsense_qa' in task_name:
        labels = []
        for d in decoded:
            patterns = {
            "the answer is": r"the\s+answer\s+is\s+\(?(\w)\)?",
        "the correct answer is": r"the\s+correct\s +answer\s+is\s+\(?(\w)\)?",
        "the answer should be" : r"the\s+answer\s+should\s be\s+\(?(\w)\)?",
        "the answer musr be" : r"the\s+answer\s+must\s be\s+\(?(\w)\)?",
        "the answer to the question is": r"the\s+answer\s+to\s+the\s+question\s+is\s+\(?(\w)\)?",
        "answer": r"answer:\s+\(?(\w)\)?"
            }
            matches = []
            for phrase, pattern in patterns.items():
                d = d.replace(',','').lower()
                d = d.replace('*','').lower()
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
    elif 'gsm8k' in task_name:
        answers = []
        for d in decoded:
            numeric_pattern = r'-?\d+\.?\d*'
            numeric_tokens = re.findall(numeric_pattern, d)
            pred = numeric_tokens[-1] if len(numeric_tokens) != 0 else ""
            if pred != "" and pred[-1] == ".":
                pred = pred[:-1]
            answers.append(pred)
        if len([x for x in answers if len(x) != 0]) > 0:
            answers = [x for x in answers if len(x) != 0]
        majority_label = max(Counter(labels), key=Counter(labels).get)
    return majority_label