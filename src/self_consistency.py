
from datetime import datetime
import json
import logging
import copy
import os
import random
import time
import shutil
from collections import defaultdict

import torch
from transformers import set_seed
from tqdm import tqdm
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5EncoderModel

from llm_wrapper import LLMWrapper
from src.utils.generation import default_metrics, \
    default_answer_prefix, default_output_prefix,\
        construct_args_from_example
import torch
from src.utils.helpers import setup_logger
from src.utils.arguments import Arguments

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
class SelfConsistency(LLMWrapper):
    def __init__(self, model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
                 is_chat=None, is_llama2=None, load_in_8bit=False):
        super().__init__(model, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage, device_map=device_map,
                         is_chat=is_chat, is_llama2=is_llama2, load_in_8bit=load_in_8bit)

        self.default_fwd_instruction = "Answer the following question with Yes or No, that is the last line of your answer should be of the format : So the answer is <your_answer>."
        self.default_fwd_question_prefix = "Question: "
        self.default_fwd_answer_prefix = "Answer: "
        
        self.default_fwd_instruction = "Answer the following question"
        self.default_fwd_input_prefix = "Question: "
        self.default_fwd_target_prefix = "Answer: "
        self.default_fwd_target_token  = "Answer"
        
        self.default_metrics = default_metrics
        if args.dataset_name == 'strategy_qa':
            # extracting the few shot exemplars for strategy_qa
            demo_file_path = '/work/pi_dhruveshpate_umass_edu/aparashar_umass_edu/Arithmetic-Sampling/data/demos/strategy_qa_demos.json'
            with open(demo_file_path, 'r') as f:
                self.datasets['demo'] = json.load(f)
            filepath = "/work/pi_dhruveshpate_umass_edu/aparashar_umass_edu/Arithmetic-Sampling/data/stratqa_data.json"
            with open(filepath,'r') as f:
                strat_data = json.load(f)['examples']
            demo_questions = [d['question'] for d in self.datasets['demo']]
            strat_data = [i for i in strat_data if i['input'] not in demo_questions]
            self.datasets[args.eval_split]   = strat_data
            breakpoint()
        
    def eval(self, inf_fn_key="zeroshot", split="dev", metrics=None, n_samples=None, task_name = None,
             dataset_sample_strategy='static', dataset_name  = None,dataset_subname = None,\
             output_sampling_strategy='max',
              verbose=False, out_dir=None, n_shots=None,
             retrieval_strategy=None, run_id=str(int(time.time())), **inf_fn_kwargs):

        #breakpoint()
        if metrics is None:
            metrics = copy.deepcopy(self.default_metrics)
        eval_args = copy.deepcopy(locals())
        del eval_args["self"]
        logger.info(f"Eval arguments: {eval_args}")
        # breakpoint()
        inf_fn = {
            'zeroshot': self.zero_shot,
            'fewshot': self.few_shot
        }[inf_fn_key]
        # breakpoint()
        dataset = self.datasets[split]
        if (n_samples is not None and n_samples != -1) and n_samples < len(dataset):
            if dataset_sample_strategy == 'static':
                dataset = list(dataset)[:n_samples]
            elif dataset_sample_strategy == 'random':
                dataset = random.sample(list(dataset), n_samples)

        references, _examples = [], []
        # Get LLM generations
        start_time = time.time()
        # breakpoint()
        _strats = ['arithmetic','temperature','topk','nucleus','eta', 'epsilon'] if output_sampling_strategy == 'all' else [
                output_sampling_strategy]
        predictions, examples = defaultdict(list), defaultdict(list)
        metric_results, results = defaultdict(dict), defaultdict(dict)
        
        for _strat in _strats:
        
            for idx, d in enumerate(tqdm(dataset, desc="Forward predicting")):
                #breakpoint()
                # try:
                inf_args, ref = construct_args_from_example(d, task_name)
                breakpoint()
                # except ValueError:
                #     logger.info('`sample_answer` not found in example while using use_answer=True mode')
                #     logger.info('Exiting...')
                #     exit()
                # Make LLM call

            end_time = time.time()
            generation_time = end_time - start_time

            
        # Select prediction from generations
        
        for _strat in _strats:
            logger.info(f"Sampling generations (strategy={_strat}):")
            start_time = time.time()
            for _ex in tqdm(_examples, desc=f"Sampling ({_strat})"):
                ex = copy.deepcopy(_ex)
                pred_cands, pred_seq_scores = zip(*ex["prediction_candidates_max"])
                if _strat == 'max':
                    pred_selected = pred_cands[0]  # the decoded list is sorted
                elif _strat == 'random':
                    llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = random.choice(pred_cands)
                elif 'arithmetic' in _strat:
                    #prepare kwargs for the sampling strategy
                    inf_args.update({
                        "question_prefix" : self.default_fwd_question_prefix,
                        "answer_prefix": self.default_fwd_answer_prefix,
                        "instructions": self.default_fwd_instruction,
                        "n_shots" : 5,
                        "demos_split":'train'
                    })
                    inf_fn_kwargs.update({
                        "do_sample": True,
                        "use_arithmetic": True,
                    })
                    
                    llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = inf_fn(**inf_args, **inf_fn_kwargs)
                    breakpoint()
                elif _strat=='eta':
                    #prepare kwargs for the sampling strategy
                    llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = inf_fn(**inf_args, **inf_fn_kwargs)
                    
                elif _strat=='epsilon':
                    #prepare kwargs for the sampling strategy
                    llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = inf_fn(**inf_args, **inf_fn_kwargs)
                
                elif 'temperature' in _strat:  
                    #prepare kwargs for the sampling strategy
                    llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = inf_fn(**inf_args, **inf_fn_kwargs)
                elif 'topk' in _strat:
                    #prepare kwargs for the sampling strategy
                    inf_fn(**inf_args, **inf_fn_kwargs)
                elif 'nucleus' in _strat:  
                    #prepare kwargs for the sampling strategy
                    inf_fn(**inf_args, **inf_fn_kwargs)
                else:
                    raise ValueError()
                ex["prediction"] = pred_selected
                predictions[_strat].append(pred_selected)
                examples[_strat].append(ex)
            end_time = time.time()
            llm_decoded = fix_posthoc(llm_decoded)
            if verbose:
                logger.info(f"Example #{idx + 1}:")
                logger.info(f"Prompt:\n{llm_prompt}")
                logger.info(f"Gold: {ref}")
                logger.info(f"Predictions: {llm_decoded}")
            if ref is not None:
                references.append(ref.lower())

           
            _examples.append({
                "idx": idx + 1,
                # "id": d["concept_set_idx"],
                "prompt": llm_prompt,
                "input": inf_args['input'],
                "reference": references[-1] if len(references) > 0 else None,
                "prediction": None,
                "prediction_candidates": llm_decoded,
            })
            # Compute metrics
            if len(references) > 0:
                for metric in metrics:
                    scores = self.get_metric_scores(metric, predictions[_strat], references)
            
                    for k in metric["score_keys"]:
                        #breakpoint()
                        metric_results[_strat][f"{metric['name']}.{k}"] = round(np.mean(scores[k]), 4)
                if verbose:
                    logger.info(metric_results[_strat])
            # Save results
            res_dir = ["eval"]
            res_dir += [self.model_name + ("_chat" if self.is_chat else ""), dataset_name,dataset_subname,split]
            if n_samples is not None:
                res_dir += [str(n_samples)]
            res_dir += [inf_fn_key, _strat]
            dt_string  = datetime.now().strftime("%d_%m_%H_%M")
            res_dir += [dt_string]
            res_dir += [run_id]
            res_dir = f"{'_'.join(res_dir)}"
            # breakpoint()
            results[_strat] = {
                "experiment": res_dir,
                "n_total": len(dataset),
                "eval_args": eval_args,
                "scores": metric_results[_strat],
                "time_taken": {
                    "generation": generation_time,
                    "sampling": end_time - start_time,
                    "total": generation_time + (end_time - start_time)
                },
                "examples": examples[_strat]
            }
            if out_dir is not None:
                res_dir_fpath = os.path.join(out_dir, res_dir)
                os.makedirs(res_dir_fpath, exist_ok=True)
                out_fname = "results.json"
                out_fpath = os.path.join(res_dir_fpath, out_fname)
                # breakpoint()
                with open(out_fpath, 'w') as fh:
                    fh.write(json.dumps(results[_strat], indent=2))
                # breakpoint()
                logger.info(f"Saved results to {out_fpath}")
            #breakpoint()
        return results
if __name__ == '__main__':
    # Setup
    cli_args = Arguments(groups=["llm", "self_consistency"])
    global args
    args = cli_args.parse_args()
    global RUN_ID
    RUN_ID = str(int(time.time())) if args.run_id is None else str(args.run_id)
    setup_logger(RUN_ID)
    logger.info("Script arguments:")
    logger.info(args.__dict__)
    set_seed(args.seed)
    # Create output dir
    out_dir = os.path.join("outputs",args.out_dir,args.dataset_name.split(':')[0])
    os.makedirs(out_dir, exist_ok=True)

    llm = SelfConsistency(model=args.model, is_chat=args.is_chat, load_in_8bit=args.load_in_8bit)
    dataset_name = args.dataset_name.split(':')[0]
    # dataset_subname = args.dataset_name.split(':')[1]
    #breakpoint()
    # llm.load_hf_data_set(split=args.eval_split,dataset_name=dataset_name,dataset_subname='')
    metrics =[{"name": "sacrebleu", 'score_keys': ['sacrebleu'], 'args': {}},{"name": "meteor", 'score_keys': ['meteor'], 'args': {}}]
    inf_fn_kwargs = {
                    "max_new_tokens": 100,
                    "do_sample": False,  # enable sampling
                    "top_p": args.eval_output_top_p,  # nucleus sampling
                    "temperature": args.eval_output_temperature,  # lower makes the distribution sharper
                    "top_k": args.eval_output_top_k, # restrict to top-k probability tokens
                    "num_beams": args.eval_output_beam_size ,  # beam search
                    "num_return_sequences": args.eval_n_samples,  # number of beams to return
                    }
    results = llm.eval(inf_fn_key=args.eval_inf_fn_key, 
                       split=args.eval_split, 
                       metrics = [k for k in default_metrics if k['name'] == 'accuracy'],
                       n_samples=args.eval_n_samples,
                       task_name=dataset_name,
                       dataset_name  = dataset_name,
                    #    dataset_subname = dataset_subname,
                       out_dir = out_dir,
                       verbose = args.verbose,
                       retrieval_strategy=args.eval_retrieval_strategy,
                       output_sampling_strategy=args.eval_output_sampling_strategy, 
                       run_id=RUN_ID,
                       dataset_sample_strategy = args.dataset_sample_strategy,
                       
                    
                       )

    if args.debug:
        breakpoint()
    
