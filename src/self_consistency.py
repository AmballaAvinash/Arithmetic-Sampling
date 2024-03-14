
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

from src.llm_wrapper import LLMWrapper
from src.utils.generation import default_metrics,fix_posthoc, \
    default_instructions, default_instructions_answer, default_fwd_question_prefix, default_inv_answer_prefix, \
    default_answer_prefix, default_fwd_instruction, default_output_prefix, default_output_prefix_chat, \
        construct_args_from_example,default_fwd_target_token,default_inv_target_token, \
    default_inv_instructions, default_inv_question_prefix, \
        default_fwd_target_prefix,default_inv_target_prefix,default_mt_fwd_target_prefix, \
 get_logprob_score, get_logprob_score_encoder, get_logprob_score_from_gen, to_tokens_and_logprobs, to_tokens_and_logprobs_pmi,get_inverse_masked_scores
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

        

        self.default_fwd_instruction = "Answer the following question"
        self.default_fwd_input_prefix = "Question: "
        self.default_fwd_target_prefix = "Answer: "
        self.default_fwd_target_token  = "Answer"
        
        self.default_metrics = default_metrics
        
    def eval(self, inf_fn_key="zeroshot", split="dev", metrics=None, n_samples=None, task_name = None,
             dataset_sample_strategy='static', dataset_name  = None,dataset_subname = None,\
             output_sampling_strategy='max',
             use_gen_norm_seq_scores=False, use_alt_seq_scores=False, verbose=False, out_dir=None, n_shots=None,
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
        dataset = self.datasets[split]['translation']
        if (n_samples is not None and n_samples != -1) and n_samples < len(dataset):
            if dataset_sample_strategy == 'static':
                dataset = list(dataset)[:n_samples]
            elif dataset_sample_strategy == 'random':
                dataset = random.sample(list(dataset), n_samples)

        references, _examples = [], []
        # Get LLM generations
        start_time = time.time()
        # breakpoint()
        for idx, d in enumerate(tqdm(dataset, desc="Forward predicting")):
            #breakpoint()
            try:
                inf_args, ref = construct_args_from_example(d, task_name)
                inf_args['input_prefix'] = inf_args['input_prefix']
                inf_args['output_prefix'] = inf_args['output_prefix']
                inf_args['instructions'] = inf_args['instructions']
                
            except ValueError:
                logger.info('`sample_answer` not found in example while using use_answer=True mode')
                logger.info('Exiting...')
                exit()
            # Make LLM call
            llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = inf_fn(**inf_args, **inf_fn_kwargs)
            # breakpoint()
            llm_seq_scores = llm_outputs.sequences_scores.tolist() if not use_gen_norm_seq_scores \
                else get_logprob_score_from_gen(llm_outputs, llm_prompt, self.model, self.tokenizer,
                                                llm_decoding_args.get('length_penalty', 1.),
                                                use_alt=use_alt_seq_scores)
            #breakpoint()
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
                "prediction_candidates_max": sorted(list(zip(llm_decoded, llm_seq_scores)), key=lambda x: x[1],
                                                    reverse=True),
                # "inverse_masked" : {"ts_strat1":ts_strat1,"ts_strat2":ts_strat2,"ts_strat3":ts_strat3,"ts_strat4":ts_strat4,"ts_strat5":ts_strat5}
                
            })

        end_time = time.time()
        generation_time = end_time - start_time

        _strats = ['eta', 'epsilon','arithmetic','temperature','topk','nucleus'] if output_sampling_strategy == 'all' else [
            output_sampling_strategy]
        predictions, examples = defaultdict(list), defaultdict(list)
        metric_results, results = defaultdict(dict), defaultdict(dict)
        # Optionally evaluate all intermediate ([0,1] in steps of 0.1) values of inverse-consistency alpha
        
        
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
                    pred_selected = random.choice(pred_cands)
                elif _strat=='eta':
                    #prepare kwargs for the sampling strategy
                    inf_fn(**inf_args, **inf_fn_kwargs)
                    
                elif _strat=='epsilon':
                    #prepare kwargs for the sampling strategy
                    inf_fn(**inf_args, **inf_fn_kwargs)
                elif 'arithmetic' in _strat:
                    #prepare kwargs for the sampling strategy
                    inf_fn(**inf_args, **inf_fn_kwargs)
                elif 'temperature' in _strat:  
                    #prepare kwargs for the sampling strategy
                    inf_fn(**inf_args, **inf_fn_kwargs)
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

            # Compute metrics
            if len(references) > 0:
                for metric in metrics:
                    #breakpoint()
                    #if metric['name'] == "coverage":
                    scores = self.get_metric_scores(metric, predictions[_strat], references)
                    #breakpoint()
                    #else:
                    #    scores = self.get_metric_scores(metric, res[_strat], gts[_strat])
                    for k in metric["score_keys"]:
                        #breakpoint()
                        metric_results[_strat][f"{metric['name']}.{k}"] = round(np.mean(scores[k]), 4)
                    #breakpoint()
                #breakpoint()
                
                if verbose:
                    logger.info(metric_results[_strat])
            # breakpoint()
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
            # check_type(results)
            # breakpoint()
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
    out_dir = os.path.join("outputs",args.out_dir,args.dataset_name.split(':')[1])
    os.makedirs(out_dir, exist_ok=True)

    llm = SelfConsistency(model=args.model, is_chat=args.is_chat, load_in_8bit=args.load_in_8bit)
    dataset_name = args.dataset_name.split(':')[0]
    dataset_subname = args.dataset_name.split(':')[1]
    #breakpoint()
    llm.load_hf_data_set(split=args.eval_split,dataset_name=dataset_name,dataset_subname=dataset_subname)
    metrics =[{"name": "sacrebleu", 'score_keys': ['sacrebleu'], 'args': {}},{"name": "meteor", 'score_keys': ['meteor'], 'args': {}}]
    results = llm.eval(inf_fn_key=args.eval_inf_fn_key, 
                       split=args.eval_split, 
                       metrics = metrics,
                       n_samples=args.eval_n_samples,
                       task_name="machinetranslation",
                       dataset_name  = dataset_name,
                       dataset_subname = dataset_subname,
                       src_lang = src_lang,
                       tgt_lang = tgt_lang,
                       out_dir = out_dir,
                       verbose = args.verbose,
                       retrieval_strategy=args.eval_retrieval_strategy,
                       output_sampling_strategy=args.eval_output_sampling_strategy, 
                       run_id=RUN_ID,
                       inv_consistency_alpha=args.eval_inv_consistency_alpha,
                       dataset_sample_strategy=args.dataset_sample_strategy,
                       all_inv_consistency_alpha=args.eval_all_inv_consistency_alpha)

    if args.debug:
        breakpoint()
    
