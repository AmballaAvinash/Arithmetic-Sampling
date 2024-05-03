
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
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5EncoderModel, LogitsProcessor,LogitsProcessorList

from llm_wrapper import LLMWrapper
from src.utils.generation import TruncateLogitsProcessor, default_metrics, \
    default_answer_prefix, default_output_prefix,default_decoding_args,\
        construct_args_from_example, fix_posthoc, get_logprob_score
import torch
from src.utils.helpers import setup_logger
from src.utils.arguments import Arguments

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
class MachineTranslation(LLMWrapper):
    def __init__(self, model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
                 is_chat=None, is_llama2=None, load_in_8bit=False):
        super().__init__(model,task_logits_processors={}, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage, device_map=device_map,
                         is_chat=is_chat, is_llama2=is_llama2, load_in_8bit=load_in_8bit)

        
        
        # self.default_fwd_instruction = "Answer the following question"
        # self.default_fwd_input_prefix = "Question: "
        # self.default_fwd_target_prefix = "Answer: "
        # self.default_fwd_target_token  = "Answer"
        
        self.default_metrics = default_metrics
        if "wmt14" in  args.dataset_name:
            # extracting the few shot exemplars for strategy_qa
            # demo_file_path = '/work/pi_dhruveshpate_umass_edu/aparashar_umass_edu/Arithmetic-Sampling/data/demos/strategy_qa_demos.json'
            # with open(demo_file_path, 'r') as f:
            #     self.datasets['demo'] = json.load(f)
            filepath = "/work/pi_dhruveshpate_umass_edu/aparashar_umass_edu/Arithmetic-Sampling/data/wmt14_fr-en.json"
            with open(filepath,'r') as f:
                mt_data = json.load(f)
            # demo_questions = [d['input'] for d in self.datasets['demo']]
            # strat_data = [i for i in strat_data if i['input'] not in demo_questions]
            self.datasets[args.eval_split]   = mt_data
        elif 'commonsense_qa' in args.dataset_name:
            demo_file_path = '/work/pi_dhruveshpate_umass_edu/aparashar_umass_edu/Arithmetic-Sampling/data/demos/commonsense_qa_demos.json'
            with open(demo_file_path, 'r') as f:
                self.datasets['demo'] = json.load(f)
    def eval(self, inf_fn_key="zeroshot", split="dev", metrics=None, n_samples=None, task_name = None,
             dataset_sample_strategy='static', dataset_name  = None,dataset_subname = None,\
             output_sampling_strategy='max', eval_dataset_size=None,eval_dataset_range=None,\
              verbose=False, out_dir=None, n_shots=None,
             retrieval_strategy=None, run_id=str(int(time.time())), **inf_fn_kwargs):

        # breakpoint()
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
        source = "French"
        target = "English"
        src_lang = "fr"
        tgt_lang = "en"
        dataset_start_point = int(eval_dataset_range.split(':')[0])
        dataset_end_point = int(eval_dataset_range.split(':')[1])
        dataset = self.datasets[split]
        if (eval_dataset_size is not None and eval_dataset_size != -1) and eval_dataset_size < len(dataset):
            if dataset_sample_strategy == 'static':
                dataset = list(dataset)[dataset_start_point:dataset_end_point]
                # breakpoint()
            elif dataset_sample_strategy == 'random':
                dataset = random.sample(list(dataset), eval_dataset_size)
        # breakpoint()
        # Get LLM generations
        if 'strategy_qa' in task_name:
            # print("no logits processor")
            inf_fn_kwargs.update({"logits_processor": self.task_logits_processors['strat_qa']})
        elif 'commonsense_qa' in task_name:
            inf_fn_kwargs.update({"logits_processor": self.task_logits_processors['strat_qa']})

        # breakpoint()
        # breakpoint()
        _strats = ['greedy','arithmetic','sample','greedy2'] if output_sampling_strategy == 'all' else [
                output_sampling_strategy]
        predictions, examples = defaultdict(list), defaultdict(list)
        metric_results, results = defaultdict(dict), defaultdict(dict)
        
        for _strat in _strats:
            references, _examples = [], []
            start_time = time.time()
            for idx, d in enumerate(tqdm(dataset, desc="Forward predicting")):
                # breakpoint()
                # try:

                inf_args, ref = construct_args_from_example(d, task_name)
                inf_args['input_prefix'] = inf_args['input_prefix'].format(source=source)
                inf_args['output_prefix'] = inf_args['output_prefix'].format(target=target)
                inf_args['instructions'] = inf_args['instructions'].format(source=source,target=target)
                inf_args['input'] = d[src_lang]
                ref = d[tgt_lang]
                # breakpoint()
                _ex = d
                logger.info(f"Sampling generations (strategy={_strat}):")
                start_time = time.time()
                # for _ex in tqdm(_examples, desc=f"Sampling ({_strat})"):
                ex = copy.deepcopy(_ex)
                if _strat == 'greedy':
                    inf_fn_kwargs.update({
                        "num_return_sequences" :  1,
                        "do_sample": False,
                    })
                    llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = inf_fn(**inf_args, **inf_fn_kwargs)  # the decoded list is sorted
                    breakpoint()
                    prediction = fix_posthoc(llm_decoded,task_name=task_name)
                elif _strat == 'greedy2':
                    inf_fn_kwargs.update({
                        "num_return_sequences" :  1,
                        "do_sample": False,
                    })
                    llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = inf_fn(**inf_args, **inf_fn_kwargs)  # the decoded list is sorted
                    # breakpoint()
                    llm_prompt_2  = llm_prompt.strip(' ').strip("Reasoning:") + " Answer: "
                    option_fwd_scores = get_logprob_score(target=inf_args["option_texts"],
                                                prefix=[llm_prompt_2]*len(inf_args["options"]),
                                                model=self.model,
                                                tokenizer=self.tokenizer,
                                                len_norm=True)
                    option_fwd_scores = [{'target':x[0],'score':x[1]} for x in zip(inf_args["option_texts"],option_fwd_scores)]
                    forward_options_sorted = sorted(option_fwd_scores, key=lambda d: d['score'],reverse= True)
                    prediction_text = forward_options_sorted[0]['target']
                    prediction = inf_args["options"][inf_args["option_texts"].index(prediction_text)].lower()

                    breakpoint()
                    # forward_pred_option = forward_options_sorted[0]['target']
                    # prediction = fix_posthoc(llm_decoded,task_name=task_name)
                elif 'arithmetic' in _strat:
                    #prepare kwargs for the sampling strategy
                    inf_fn_kwargs.update({
                        "do_sample": True,
                        "use_arithmetic": True,
                        "num_return_sequences" : args.eval_n_samples
                    })
                    # try:
                    llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = inf_fn(**inf_args, **inf_fn_kwargs)
                    prediction = fix_posthoc(llm_decoded,task_name=task_name)
                    # breakpoint()
                    # except:
                    #     continue
                elif _strat=='sample':
                    #prepare kwargs for the sampling strategy
                    # breakpoint()
                    inf_fn_kwargs.update({
                        "do_sample": True,
                        "num_return_sequences" : args.eval_n_samples,
                        "use_arithmetic": False,
                    })
                    llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = inf_fn(**inf_args, **inf_fn_kwargs)
                    prediction = fix_posthoc(llm_decoded,task_name=task_name)
                    # breakpoint()

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
                predictions[_strat].append(prediction)
                if ref is not None:
                    references.append(ref.lower())
                ex.update({
                    "idx": idx + 1,
                    # "id": d["concept_set_idx"],
                    "prompt": llm_prompt,
                    "input": inf_args['question'],
                    "reference": references[-1] if len(references) > 0 else None,
                    "prediction": prediction,
                    "generation_candidates": llm_decoded,
                    "n_gram_diversity" : self.compute_ngram_diversity(llm_decoded)
                })
                examples[_strat].append(ex)
            
                if verbose:
                    logger.info(f"Example #{idx + 1}:")
                    logger.info(f"Prompt:\n{llm_prompt}")
                    logger.info(f"Gold: {ref}")
                    logger.info(f"Predictions: {llm_decoded}")
                

            # breakpoint()
           
            
            end_time = time.time()
            for _ex in examples[_strat]:
                _ex["n_gram_diversity"] = self.compute_ngram_diversity(_ex["generation_candidates"])
                _ex["duplicity"] = self.compute_duplicity(_ex["generation_candidates"])
            # Compute metrics
            if len(references) > 0:
                for metric in metrics:
                    # breakpoint()
                    scores = self.get_metric_scores(metric, predictions[_strat], references)
            
                    for k in metric["score_keys"]:
                        #breakpoint()
                        metric_results[_strat][f"{metric['name']}.{k}"] = round(np.mean(scores[k]), 4)
                if verbose:
                    logger.info(metric_results[_strat])
            # breakpoint()
            metric_results[_strat]["n_gram_diversity"] = np.mean([_ex["n_gram_diversity"] for _ex in examples[_strat]])
            metric_results[_strat]["duplicity"] = np.mean([_ex["duplicity"] for _ex in examples[_strat]])
            # Save results
            res_dir = ["eval"]
            res_dir += [self.model_name + ("_chat" if self.is_chat else ""), dataset_name,dataset_subname,split]
            if eval_dataset_size is not None:
                res_dir += [str(eval_dataset_size)]
            res_dir += [inf_fn_key, _strat,str(inf_fn_kwargs["num_return_sequences"]),str(dataset_start_point),"to",str(dataset_end_point)]
            dt_string  = datetime.now().strftime("%d_%m_%H_%M")
            res_dir += [dt_string]
            res_dir += [run_id]
            res_dir = f"{'_'.join(res_dir)}"
            # breakpoint()
            llm_decoding_args_to_save = {k:v for k,v in llm_decoding_args.items() if k != "logits_processor"}
            results[_strat] = {
                "experiment": res_dir,
                "n_total": len(dataset),
                "eval_args": llm_decoding_args_to_save,
                "scores": metric_results[_strat],
                "time_taken": {
                    "sampling": end_time - start_time,
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

    llm = MachineTranslation(model=args.model, is_chat=args.is_chat, load_in_8bit=args.load_in_8bit)
    dataset_name = args.dataset_name.split(':')[0]
    # dataset_subname = args.dataset_name.split(':')[1]
    #breakpoint()
    if 'commonsense_qa' in args.dataset_name:
        llm.load_hf_data_set(split=args.eval_split,dataset_name=dataset_name,dataset_subname='')
    metrics =[{"name": "bleu", 'score_keys': ['bleu'], 'args': {}},{"name": "ngram_diversity", 'score_keys': ['ngram_diversity'], 'args': {}}]
    inf_fn_kwargs = {
                    "max_new_tokens": 200,
                    "do_sample": False,  # enable sampling
                    "top_p": args.eval_output_top_p if args.eval_output_top_p is not None else default_decoding_args["eval_output_top_p"],  # nucleus sampling
                    "temperature": args.eval_output_temperature if args.eval_output_temperature is not None else default_decoding_args["temperature"],  # lower makes the distribution sharper
                    "top_k": args.eval_output_top_k if args.eval_output_top_k is not None else default_decoding_args["top_k"], # restrict to top-k probability tokens
                    "num_beams": args.eval_output_beam_size if args.eval_output_beam_size is not None else default_decoding_args["num_beams"], # beam search
                    "num_return_sequences": args.eval_n_samples,  # number of beams to return
                    }
    results = llm.eval(inf_fn_key=args.eval_inf_fn_key, 
                       split=args.eval_split, 
                       metrics = metrics,
                       eval_dataset_size = args.eval_dataset_size,
                       n_samples=args.eval_n_samples,
                       task_name=dataset_name,
                       dataset_name  = dataset_name,
                       dataset_subname = '',
                       eval_dataset_range = args.eval_dataset_range,
                       out_dir = out_dir,
                       verbose = args.verbose,
                       retrieval_strategy=args.eval_retrieval_strategy,
                       output_sampling_strategy=args.eval_output_sampling_strategy, 
                       run_id=RUN_ID,
                       dataset_sample_strategy = args.dataset_sample_strategy,
                       **inf_fn_kwargs
                       )

    if args.debug:
        print('done')
        # breakpoint()
    
