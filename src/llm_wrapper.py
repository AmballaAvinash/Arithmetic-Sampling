from collections import defaultdict
import json
import logging
import copy
import os
from pathlib import Path
import random
import numpy as np
import datasets
from datasets import load_dataset
import torch
from sacrebleu.metrics import BLEU
import nltk
from nltk.translate import meteor
from nltk import word_tokenize
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, GPT2Tokenizer,GPT2LMHeadModel,\
    T5ForConditionalGeneration
from src.utils.generation import construct_prompt_from_args,construct_qa_prompt_from_args, default_metrics, default_decoding_args, \
     default_input_prefix,\
    default_answer_prefix, default_output_prefix, \
    TruncateLogitsProcessor, prompt_arr_2_text \

    
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMWrapper:
    def __init__(self, model=None, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
                 is_chat=None, is_llama2=None, is_t5=None, is_mpt=None,is_bloom=None,is_gpt2_mid= None,is_gemma =None, load_in_8bit=False):
        self.model_name = None
        self.model = None
        self.tokenizer = None
        self.is_llama2 = is_llama2
        self.is_t5 = is_t5
        self.is_gemma = is_gemma
        self.is_chat = is_chat
        if model is not None:
             self.model_name = model.lower().split('/')[-1]   # then infer from name
        if self.is_llama2 is None:  
            self.is_llama2 = 'llama-2' in self.model_name
        if self.is_t5 is None:  
            self.is_t5 = 't5' in self.model_name
        if self.is_gemma is None:  
            self.is_gemma = 'gemma' in self.model_name
        cls_tokenizer = AutoTokenizer
        tokenizer_args = {}
        cls_model = AutoModelForCausalLM
        if self.is_llama2:
            cls_tokenizer = LlamaTokenizer
            tokenizer_args.update({"add_bos_token": True, "add_eos_token": False})
            cls_model = LlamaForCausalLM
            # cls_tokenizer.pad_token = cls_tokenizer.eos_token
            # cls_model.config.pad_token_id = cls_model.config.eos_token_id
        self.tokenizer = cls_tokenizer.from_pretrained(model, **tokenizer_args)
        if self.is_t5:
            cls_model = T5ForConditionalGeneration
            torch_dtype = torch.float32  # because of a logsoftmax error with half precision; TODO: double-check
        if self.is_gemma:
            cls_model = AutoModelForCausalLM
            cls_tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b',cache_dir = '')
            
        self.model = cls_model.from_pretrained(model,
                                                torch_dtype=torch_dtype,
                                                device_map=device_map,
                                                # low_cpu_mem_usage=low_cpu_mem_usage,
                                                cache_dir = '/work/pi_mccallum_umass_edu/aparashar_umass_edu/models/.cache',
                                                trust_remote_code=True,
                                                load_in_8bit=load_in_8bit)
        self.tokenizer.pad_token =  self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.eval()

        self.default_decoding_args = default_decoding_args
        if any([m in self.model_name for m in ['falcon', 'mpt']]):
            self.default_decoding_args.update({
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id
            })
        self.strip_prompt = any([m in self.model_name for m in ['falcon']])
        # Add stopping criterion that terminates generation when a newline is generated
        
        # TODO: Change these defaults to be generic (not task-specific)
        # self.default_instructions = default_instructions
        # self.default_instructions_answer = default_instructions_answer
        self.default_answer_prefix = default_answer_prefix
        self.default_question_prefix = default_output_prefix  # Used in few-shot demonstrations
        self.default_output_prefix = default_output_prefix
        self.default_metrics = default_metrics
        self.datasets = {
            'train': None,
            'dev': None,
            'test': None
        }
        self.dataset_fpaths = {
            'train': None,
            'dev': None,
            'test': None
        }
        self.metrics = {}
    def load_hf_data_set(self,split,dataset_name, dataset_subname):
        # breakpoint()
        datasets.config.DOWNLOADED_DATASETS_PATH = Path('/work/pi_mccallum_umass_edu/aparashar_umass_edu/datasets')
        self.datasets[split] = datasets.load_dataset(dataset_name,dataset_subname, split="validation",cache_dir ='/work/pi_mccallum_umass_edu/aparashar_umass_edu/datasets/.cache' )
        # breakpoint()
    def _base_generator(self, prompt, return_output_after_prompt=True, **kwargs):
        if self.is_t5:
            return_output_after_prompt = False

        _prompt = prompt

        decoding_args = copy.deepcopy(self.default_decoding_args)
        decoding_args.update(kwargs)

        if self.strip_prompt:
            _prompt = _prompt.strip()

        assert _prompt != ""

        # Tokenize
        prompt_tokenized = self.tokenizer(_prompt, return_tensors="pt", return_token_type_ids=False)
        prompt_tokenized.to("cuda")
        if decoding_args.get('logits_processor', None) is not None:
            # Set prompt length for logits processor
            decoding_args['logits_processor'][0].set_prompt_len(prompt_tokenized.input_ids.size(1))

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**prompt_tokenized, **decoding_args,
                                          return_dict_in_generate=True, output_scores=True)
        decoded = [
            self.tokenizer.decode(o, skip_special_tokens=True)[(len(_prompt) if return_output_after_prompt else 0):] for
            o in outputs.sequences]
        return decoded, outputs, _prompt, decoding_args

    def zero_shot(self, input=None, input_prefix=None, output_prefix=None, instructions=None, prompt_sep="\n",task=None, **kwargs):
        # Construct prompt
        prompt = ""
        prompt_arr = []
        # Task instructions
        if instructions is None:
            instructions = copy.copy(self.default_instructions)
        if instructions != "":
            prompt_arr.append(instructions)
        prompt_arr += construct_prompt_from_args(input, input_prefix)
        # Get prompt text
        prompt = prompt_arr_2_text(prompt_arr, prompt_sep, self.is_llama2, self.is_chat,
                                   self.default_output_prefix if output_prefix is None else output_prefix)

        return self._base_generator(prompt, **kwargs)
    def few_shot(self, question, construct_args_fn, question_prefix=None, answer=None, answer_prefix=None,
                instructions=None, output_prefix=None, prompt_sep="\n\n", n_shots=1,
                 retrieval_strategy='random', demos_split='train',use_answer=False, **kwargs):
        demos = self.datasets[demos_split]
        # more retrieval strategies can be added if required
        if retrieval_strategy == "random":
            sampled_demos = random.sample(demos, n_shots)
        

        # Construct prompt
        prompt = ""
        prompt_arr = []
        # Task instructions
        if instructions is None:
            instructions = copy.copy(self.default_instructions)
            if answer is not None:
                instructions += self.default_instructions_answer
            instructions = " ".join(instructions)
        if instructions != "":
            prompt_arr.append(instructions)
        # Demonstrations
        for d in sampled_demos:
            d_inf_args, d_ref = construct_args_fn(d)
            prompt_arr += construct_qa_prompt_from_args(d_inf_args['question'],self.default_question_prefix,
                                                            d_inf_args['answer'],self.default_answer_prefix)
        # Question
        prompt_arr += construct_qa_prompt_from_args(question,self.default_question_prefix)
        # Get prompt text
        prompt = prompt_arr_2_text(prompt_arr, prompt_sep, self.is_llama2, self.is_chat,
                                   self.default_output_prefix if output_prefix is None else output_prefix)

        return self._base_generator(prompt, **kwargs)
    
    def load_metrics(self, metrics):
        self.metrics.update({m: load(m) for m in metrics if m not in self.metrics})

    def  get_metric_scores(self, metric, predictions, references):
        scores = defaultdict()
        #breakpoint()
        print(metric, predictions, references)
        if metric["name"] == 'accuracy':
            # breakpoint()
            scores['accuracy'] = self.compute_accuracy(predictions=predictions,references=references)
        if metric["name"] == "sacrebleu":
            scores["sacrebleu"] = self.compute_sacrebleu(predictions,references)
        if metric["name"] == "meteor":
            scores["meteor"] = self.compute_meteor(predictions,references)
        
        return scores
    
    def compute_accuracy(self,predictions,references):
        return np.mean(list(map(lambda x:x[0]==x[1],zip(predictions,references))))
    def compute_sacrebleu(self,predictions,references):
        bleu = BLEU()
        references = [references]
        result = bleu.corpus_score(predictions,references)
        #breakpoint()
        return result.score
    def compute_meteor(self,predictions,references):
        meteors = []
        
        for i in range(len(predictions)):
            meteors.append(round(meteor([word_tokenize(references[i])],word_tokenize(predictions[i])), 4))
        #breakpoint()
        return np.mean(meteors)
