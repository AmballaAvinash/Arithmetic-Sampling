import json
import os
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import random

class ReasoningResultsAnalyzer:
    def __init__(self, sequences_filenames, evaluation_filename) -> None:
        self.sequences_filenames = sequences_filenames
        self.evaluation_filename = evaluation_filename
        self.results = {}

    def aggregrate_results(self):
        for sequences_filename in self.sequences_filenames:
            sequences = {}
            if os.path.exists(sequences_filename):
                with open(sequences_filename, "r") as sequences_file:
                    sequences = json.load(sequences_file)
            for arg_key, data in sequences.items():
                for subsample in [5, 10, 20]:
                    subsample_arg_key = arg_key+f"_subsample_{subsample}"
                    arithmetic_ngram_diversity, ancestral_ngram_diversity, greedy_ngram_diversity = 0, 0, 0
                    arithmetic_vote_accuracy, ancestral_vote_accuracy, greedy_accuracy = 0, 0, 0
                    for qa_pair in data:
                        arithmetic_ngram_diversity += ngram_diversity(qa_pair["arithmetic_sampling"], subsample, post_process_step=True)
                        ancestral_ngram_diversity += ngram_diversity(qa_pair["ancestral_sampling"], subsample, post_process_step=True)
                        greedy_ngram_diversity += ngram_diversity(qa_pair["greedy_decoding"], subsample=1, post_process_step=True)
                        arithmetic_vote_accuracy += voting_accuracy(qa_pair["arithmetic_sampling"], qa_pair["gt"], subsample, post_process_step=True)
                        ancestral_vote_accuracy += voting_accuracy(qa_pair["ancestral_sampling"], qa_pair["gt"], subsample, post_process_step=True)
                        greedy_accuracy += numerical_accuracy(qa_pair["greedy_decoding"], qa_pair["gt"], subsample=1, post_process_step=True)
                    if self.results.get(subsample_arg_key, None) is None:
                        self.results[subsample_arg_key] = {
                            "ngram_diversity": {
                                "arithmetic_sample":[arithmetic_ngram_diversity / len(data)],
                                "ancestral_sample": [ancestral_ngram_diversity / len(data)],
                                "greedy_decoding": [greedy_ngram_diversity / len(data)],
                            },
                            "accuracy": {
                                "arithmetic_sample": [arithmetic_vote_accuracy / len(data)],
                                "ancestral_sample": [ancestral_vote_accuracy / len(data)],
                                "greedy_decoding": [greedy_accuracy / len(data)],
                            }
                        }
                    else:
                        self.results[subsample_arg_key]["ngram_diversity"]["arithmetic_sample"].append(arithmetic_ngram_diversity / len(data))
                        self.results[subsample_arg_key]["ngram_diversity"]["ancestral_sample"].append(ancestral_ngram_diversity / len(data))
                        self.results[subsample_arg_key]["ngram_diversity"]["greedy_decoding"].append(greedy_ngram_diversity / len(data))
                        self.results[subsample_arg_key]["accuracy"]["arithmetic_sample"].append(arithmetic_vote_accuracy / len(data))
                        self.results[subsample_arg_key]["accuracy"]["ancestral_sample"].append(ancestral_vote_accuracy / len(data))
                        self.results[subsample_arg_key]["accuracy"]["greedy_decoding"].append(greedy_accuracy / len(data))
        with open(results_filename, "w") as results_file:
            json.dump(self.results, results_file)

    def plot_results(self, plot_filename, results_filename = None, errorbar = False):
        if results_filename is not None:
            with open(results_filename, "r") as results_file:
                self.results = json.load(results_file)
        subsampled_sequences = [5, 10, 20]
        configs = {
            'num_examples=100, num_samples=20, temperature=0.5, shots=8, top_k=40, top_p=1.0': "T = 0.5, k = 40"
        }
        for metric in ["accuracy", "ngram_diversity"]:
            for key in configs:
                for sampling in ["arithmetic_sample", "ancestral_sample"]:
                    Y, Y_err = [], []
                    for subsample in subsampled_sequences:
                        subsample_arg_key = key + f"_subsample_{subsample}"
                        arr = self.results[subsample_arg_key][metric][sampling]
                        Y.append(np.mean(arr))
                        Y_err.append(np.std(arr))
                    if errorbar:
                        plt.errorbar(subsampled_sequences, Y, yerr=Y_err, label = f"{sampling} {configs[key]}", marker='s')
                    else:
                        plt.plot(subsampled_sequences, Y, label = f"{sampling} {configs[key]}", marker='s')
            greedy_arg_key = f"num_examples=100, num_samples=20, temperature=0.5, shots=8, top_k=40, top_p=1.0_subsample_{subsample}"
            arr = self.results[greedy_arg_key][metric]["greedy_decoding"]
            Y = [np.mean(arr)]*len(subsampled_sequences)
            Y_err = [np.std(arr)]*len(subsampled_sequences)
            if errorbar:
                plt.errorbar(subsampled_sequences, Y, yerr=Y_err, label = "greedy_decode", marker='o')
            else:
                plt.plot(subsampled_sequences, Y, label = "greedy_decode", marker='o')
            plt.legend()
            plt.xlabel("#Sampled Reasoning Paths")
            plt.ylabel(metric)
            plt.xticks(range(0, 25, 5))
            plt.title(f"{'Accuracy' if metric == 'accuracy' else 'Diversity'} Comparison (100 Q/A pairs) w/ LLama-2-7B (int8)")
            plt.grid(True)
            plt.savefig(f"{plot_filename}_{metric}.png")
            plt.clf()

if __name__ == "__main__":
    # output_filenames = [f"gemma-2b-it__gsm8k__{seed}_output.json" for seed in [786, 275, 606, 402, 702]]
    # sequences_filenames = ["./outputs/week_may_03/gemma-2b_gsm8k_samples_250.json"]
    # results_filename = "./outputs/week_may_03/gemma-2b__gsm8k__results.json"
    # results_analyzer = ReasoningResultsAnalyzer(sequences_filenames, results_filename)
    # results_analyzer.aggregrate_results()
    # results_analyzer.plot_results("./outputs/week_may_03/gemma-2b", results_filename)
    # sequences_filenames = ["./outputs/week_may_03/gemma-7b_gsm8k_samples_250.json"]
    # results_filename = "./outputs/week_may_03/gemma-7b__gsm8k__results.json"
    # results_analyzer = ReasoningResultsAnalyzer(sequences_filenames, results_filename)
    # results_analyzer.aggregrate_results()
    # results_analyzer.plot_results("./outputs/week_may_03/gemma-7b", results_filename)
    sequences_filenames = ["./outputs/week_may_10/llama-2-7b_gsm8k_samples.json"]
    results_filename = "./outputs/week_may_10/llama-2-7b__gsm8k__results.json"
    results_analyzer = ReasoningResultsAnalyzer(sequences_filenames, results_filename)
    results_analyzer.aggregrate_results()
    results_analyzer.plot_results("./outputs/week_may_10/llama-2-7b", results_filename)
