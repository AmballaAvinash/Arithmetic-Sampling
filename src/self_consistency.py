
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
class MachineTranslation(LLMWrapper):
    def __init__(self, model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
                 is_chat=None, is_llama2=None, load_in_8bit=False):
        super().__init__(model, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage, device_map=device_map,
                         is_chat=is_chat, is_llama2=is_llama2, load_in_8bit=load_in_8bit)

        

        self.default_fwd_instruction = "Translate the following German sentence to English:"
        self.default_fwd_input_prefix = "German: "
        self.default_fwd_target_prefix = "English: "
        self.default_fwd_target_token  = "German"
        self.lang_store = {'de':'German','en':'English','fr':'French','fi':'Finnish'}
        
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
                inf_args['input_prefix'] = inf_args['input_prefix'].format(source=source)
                inf_args['output_prefix'] = inf_args['output_prefix'].format(target=target)
                inf_args['instructions'] = inf_args['instructions'].format(source=source,target=target)
                inf_args['input'] = d[src_lang]
                ref = d[tgt_lang]
               # breakpoint()
                #  question,_,options,answer = construct_args_from_example(d,task_name)
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

            inv_masked_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            inv_masked_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")
            inv_masked_enc_model = T5EncoderModel.from_pretrained("google/flan-t5-base")
            
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

        _strats = ['max', 'inverse-len'] if output_sampling_strategy == 'all' else [
            output_sampling_strategy]
        predictions, examples = defaultdict(list), defaultdict(list)
        metric_results, results = defaultdict(dict), defaultdict(dict)
        # Optionally evaluate all intermediate ([0,1] in steps of 0.1) values of inverse-consistency alpha
        _inv_alpha = list(np.arange(0, 11) / 10) if all_inv_consistency_alpha else []
        _inv_alpha_preds = defaultdict(list)
        inv_masked_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        inv_masked_enc_model = T5EncoderModel.from_pretrained("google/flan-t5-base")
        inv_masked_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")
        
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
                elif _strat=='pmi':
                    try:
                        cand_pmi_logprobs = [x['score'] for  x in to_tokens_and_logprobs_pmi(pred_cands,model=self.model,tokenizer=self.tokenizer)]
                        cand_pmi_scores = list(map(lambda x:x[0]-x[1],zip(pred_seq_scores,cand_pmi_logprobs)))
                        pred_pmi_cands_scored = sorted(list(zip(pred_cands, cand_pmi_scores)), key=lambda x: x[1], reverse=True)
                        #breakpoint()
                        pred_selected = pred_pmi_cands_scored[0][0]
                    except:
                        print(_ex)
                        continue
                elif _strat=='pmi_dc':
                    x_domain = default_mt_fwd_target_prefix.format(target = target)
                    #breakpoint()
                    pmi_dc_logprobs = get_logprob_score(target=pred_cands,
                                                      prefix=[x_domain]*len(pred_cands),
                                                      model=self.model,
                                                      tokenizer=self.tokenizer,
                                                      len_norm=True)
                    cand_pmi_dc_scores = list(map(lambda x:x[0]-x[1],zip(pred_seq_scores,pmi_dc_logprobs)))
                    pred_pmi_cands_scored = sorted(list(zip(pred_cands, cand_pmi_dc_scores)), key=lambda x: x[1], reverse=True)
                    pred_selected = pred_pmi_cands_scored[0][0]
                elif 'inverse_enc' in _strat:
                    ts_enc_strat = get_logprob_score_encoder(prefixes=[ex['input']] * len(pred_cands),
                                                                    targets = pred_cands,
                                                                    model = inv_masked_enc_model,
                                                                    tokenizer = inv_masked_tokenizer)
                    # breakpoint()
                    pred_cands_scored = sorted(ts_enc_strat.items(), key = lambda x: x[1],reverse=True)
                    pred_selected = pred_cands_scored[0][0]
                elif 'inverse_masked' in _strat:  # inverse-consistency
                    # breakpoint()
                    if "_strat1" in _strat:
                        pred_cands_scored = sorted(ex["inverse_masked"]["ts_strat1"].items(), key = lambda x: x[1],reverse=True)
                        pred_selected = pred_cands_scored[0][0]
                    if "_strat2" in _strat:
                        pred_cands_scored = sorted(ex["inverse_masked"]["ts_strat2"].items(), key = lambda x: x[1],reverse=True)
                        pred_selected = pred_cands_scored[0][0]
                    if "_strat3" in _strat:
                        pred_cands_scored = sorted(ex["inverse_masked"]["ts_strat3"].items(), key = lambda x: x[1],reverse=True)
                        pred_selected = pred_cands_scored[0][0]
                    if "_strat4" in _strat:
                        pred_cands_scored = sorted(ex["inverse_masked"]["ts_strat4"].items(), key = lambda x: x[1],reverse=True)
                        pred_selected = pred_cands_scored[0][0]
                    if "_strat5" in _strat:
                        pred_cands_scored = sorted(ex["inverse_masked"]["ts_strat5"].items(), key = lambda x: x[1],reverse=True)
                        pred_selected = pred_cands_scored[0][0]
                elif 'inverse_enc' in _strat:
                    ts_enc_strat = get_logprob_score_encoder(prefixes=[ex['input']] * len(pred_cands),
                                                                    targets = pred_cands,
                                                                    model = inv_masked_enc_model,
                                                                    tokenizer = inv_masked_tokenizer)
                    # breakpoint()
                    pred_cands_scored = sorted(ts_enc_strat.items(), key = lambda x: x[1],reverse=True)
                    pred_selected = pred_cands_scored[0][0]
                elif 'inverse-len' in _strat:  # inverse-consistency
                    cand_prompts = []
                    for cand_s in pred_cands:
                        inv_sentence = f"""{self.default_inv_input_prefix.format(target=target)}{cand_s}"""
                        inv_prompt = [self.default_inv_instructions.format(target=target,source=source), inv_sentence, self.default_inv_target_prefix.format(source=source)]
                        # TODO: Add schema information if use_schema == True
                        # TODO: Use chat-based prompts if using chat models
                        inv_prompt = "\n".join(inv_prompt)
                        #breakpoint()
                        cand_prompts.append(inv_prompt)
                    cand_logprobs = get_logprob_score(target=[ex['input']] * len(cand_prompts),
                                                      prefix=cand_prompts,
                                                      model=self.model,
                                                      tokenizer=self.tokenizer,
                                                      len_norm='len-norm' in _strat)
                    assert 0. <= inv_consistency_alpha <= 1.
                    cand_scores = list(map(lambda x: (1 - inv_consistency_alpha) * x[0] + inv_consistency_alpha * x[1],
                                           zip(pred_seq_scores, cand_logprobs)))
                    pred_cands_scored = sorted(list(zip(pred_cands, cand_scores)), key=lambda x: x[1], reverse=True)
                    pred_selected = pred_cands_scored[0][0]
                    ex[f"prediction_candidates_{_strat}"] = pred_cands_scored
                    for _alpha in _inv_alpha:
                        _alpha_cand_scores = list(
                            map(lambda x: (1 - _alpha) * x[0] + _alpha * x[1], zip(pred_seq_scores, cand_logprobs)))
                        _alpha_pred_cands_scored = sorted(list(zip(pred_cands, _alpha_cand_scores)), key=lambda x: x[1],
                                                          reverse=True)
                        ex[f"prediction_candidates_{_strat}_{_alpha}"] = _alpha_pred_cands_scored
                        _inv_alpha_preds[_alpha].append(_alpha_pred_cands_scored[0][0])
                    # breakpoint()
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
                if 'inverse-len' in _strat:
                    # Add scores for all intermediate alpha values, if requested
                    #breakpoint()
                    for _alpha in _inv_alpha_preds:
                        if 'all_alpha' not in metric_results[_strat]:
                            metric_results[_strat]['all_alpha'] = defaultdict(dict)
                        for metric in metrics:
                            _alpha_scores = self.get_metric_scores(metric, _inv_alpha_preds[_alpha], references)
                            for k in metric["score_keys"]:
                                metric_results[_strat]['all_alpha'][_alpha][f"{metric['name']}.{k}"] = round(
                                    np.mean(_alpha_scores[k]), 4)
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
    cli_args = Arguments(groups=["llm", "machinetranslation"])
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
    # if args.clean_out_dir:
    #     if os.path.isdir(out_dir):
    #         for f in os.listdir(out_dir):
    #             fpath = os.path.join(out_dir, f)
    #             if os.path.isdir(fpath):
    #                 shutil.rmtree(fpath)
    os.makedirs(out_dir, exist_ok=True)

    llm = MachineTranslation(model=args.model, is_chat=args.is_chat, load_in_8bit=args.load_in_8bit)
    dataset_name = args.dataset_name.split(':')[0]
    dataset_subname = args.dataset_name.split(':')[1]
    src_lang = dataset_subname.split('-')[0]
    tgt_lang = dataset_subname.split('-')[1]
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
    
