from collections import Counter
import os
import json
import random
import re
import numpy as np
import pathlib
from nltk.util import ngrams
def fix_posthoc(decoded,task_name='strategy_qa'):
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
#                 print(f"Answer couldn't be extracted: {d}")
                labels.append('no')
        if len([x for x in labels if x != 'nan']) > 0:
            labels = [x for x in labels if x != 'nan']
        majority_label = max( Counter(labels), key=Counter(labels).get)
#     else:
    elif 'commonsense_qa' in task_name:
        labels = []
        for d in decoded:
            patterns = {
            "the answer is": r"the\s+answer\s+is\s+\(?(\w)\)?",
        "the correct answer is": r"the\s+correct\s +answer\s+is\s+\(?(\w)\)?",
        "the answer should be" : r"the\s+answer\s+should\s be\s+\(?(\w)\)?",
        "the answer must be" : r"the\s+answer\s+must\s be\s+\(?(\w)\)?",
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
#                 pr/int(f"Answer couldn't be extracted: {d}")
                labels.append('nan')
        if len([x for x in labels if x != 'nan']) > 0:
            labels = [x for x in labels if x != 'nan']
        majority_label = max( Counter(labels), key=Counter(labels).get)
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
def compute_ngram_diversity(translations):
    ngram_diversity_score = 0
    n_values = [1, 2, 3, 4]
    n_gram_div_scores = []
    for n in n_values:
        unique_ngrams = set()
        total_ngram_count = 0
        for translation in translations:
            # Compute n-grams
            translation_ngrams = list(ngrams(translation.split(), n))
            # Count unique n-grams
            total_ngram_count += len(list(translation_ngrams))
            unique_ngrams.update(translation_ngrams)
        # Update total counts
        total_unique_ngrams = len(unique_ngrams)
        ngram_diversity_score += total_unique_ngrams / (total_ngram_count + torch.finfo(torch.float32).eps)
    return ngram_diversity_score
def cal_metrics(examples__s):
    n_gram_diversity = compute_ngram_diversity(examples__s['cands'])
    accuracy = 1 if examples__s["reference"] == examples__s["prediction"].lower() else 0
    return {'n_gram_diversity':n_gram_diversity,'accuracy':accuracy}     
def ele_dist_k_from_idx(lst, start_index, k):
    return lst[start_index::k]

def process_results(path, task_name, num_return_sequences):
    for file in os.listdir(pathlib.Path(path)):
        if file == 'results.json':
            f_path = pathlib.Path(path, file)
            with open(f_path, 'r') as f:
                result_json = json.load(f)
                if result_json['eval_args']['num_return_sequences'] == num_return_sequences:
                    experiment = result_json['experiment']
                    n_total = result_json['n_total']
                    eval_args = result_json['eval_args']
                    accuracy_sample = {}
                    accuracy_arith = {}

                    if not result_json['eval_args']['use_arithmetic']:
                        accuracy_samples = {}
                        ngd_samples = {}
                        sample_examples = {}
                        for iterr in range(50):
                            sample_examples[iterr] = []
                            for i, _ex in enumerate(result_json['examples']):
                                sample_cands = {}
                                for n in range(1, num_return_sequences // 2 + 1):
                                    sample_cands[n] = {}
                                    cands = _ex['generation_candidates'].copy()
                                    sample_cands[n]['cands'] = random.sample(cands, n)
                                    sample_cands[n]['reference'] = _ex['reference']
                                    sample_cands[n]['prediction'] = fix_posthoc(sample_cands[n]['cands'], task_name)
                                    sample_cands[n]['metrics'] = cal_metrics(sample_cands[n])
                                sample_examples[iterr].append(sample_cands.copy())
                            
                            accuracy_samples[iterr] = {n: np.mean([e[n]['metrics']["accuracy"] for e in sample_examples[iterr]]) for n in range(1, num_return_sequences // 2 + 1)}
                            ngd_samples[iterr] = {n: np.mean([e[n]['metrics']["n_gram_diversity"] for e in sample_examples[iterr]]) for n in range(1, num_return_sequences // 2 + 1)}
                        
                        accuracy_sample['mean'] = {n: np.mean([accuracy_samples[iterr][n] for iterr in range(50)]) for n in range(1, num_return_sequences // 2 + 1)}
                        accuracy_sample['std'] = {n: np.std([accuracy_samples[iterr][n] for iterr in range(50)]) for n in range(1, num_return_sequences // 2 + 1)}
                        accuracy_sample['mean'][num_return_sequences] = result_json["scores"]["accuracy"]
                        accuracy_sample['std'][num_return_sequences] = 0

                    elif result_json['eval_args']['use_arithmetic']:
                        accuracy_samples = {}
                        ngd_samples = {}
                        arith_examples = {}
                        for iterr in range(20):
                            arith_examples[iterr] = []
                            for i, _ex in enumerate(result_json['examples']):
                                arith_cands = {}
                                for n in range(1, num_return_sequences // 2 + 1):
                                    offset = random.choice(range(num_return_sequences // n))
                                    arith_cands[n] = {}
                                    cands = _ex['generation_candidates'].copy()
                                    arith_cands[n]['cands'] = ele_dist_k_from_idx(cands, offset, num_return_sequences // n)
                                    arith_cands[n]['reference'] = _ex['reference']
                                    arith_cands[n]['prediction'] = fix_posthoc(arith_cands[n]['cands'], task_name)
                                    arith_cands[n]['metrics'] = cal_metrics(arith_cands[n])
                                arith_examples[iterr].append(arith_cands.copy())
                            
                            accuracy_samples[iterr] = {n: np.mean([e[n]['metrics']["accuracy"] for e in arith_examples[iterr]]) for n in range(1, num_return_sequences // 2 + 1)}
                            ngd_samples[iterr] = {n: np.mean([e[n]['metrics']["n_gram_diversity"] for e in arith_examples[iterr]]) for n in range(1, num_return_sequences // 2 + 1)}
                        
                        accuracy_arith['mean'] = {n: np.mean([accuracy_samples[iterr][n] for iterr in range(20)]) for n in range(1, num_return_sequences // 2 + 1)}
                        accuracy_arith['std'] = {n: np.std([accuracy_samples[iterr][n] for iterr in range(20)]) for n in range(1, num_return_sequences // 2 + 1)}
                        accuracy_arith['mean'][num_return_sequences] = result_json["scores"]["accuracy"]
                        accuracy_arith['std'][num_return_sequences] = 0
                        
    return accuracy_sample, accuracy_arith

if __name__ == "__main__":
    path = '/Users/adityaparashar/UMass/696DS/Arithmetic-Sampling/outputs/outputs/self_consistency/commonsense_qa/eval_llama-2-7b-hf_commonsense_qa__validation_1221_fewshot_sample_40_0_to_1221_10_05_09_59_0/results.json'
    task_name = "commonsense_qa"
    num_return_sequences = 40

    accuracy_sample, accuracy_arith = process_results(path, task_name, num_return_sequences)
    
    # You can now print or use the accuracy_sample and accuracy_arith as needed
    print("Accuracy Sample:", accuracy_sample)
    print("Accuracy Arithmetic:", accuracy_arith)
