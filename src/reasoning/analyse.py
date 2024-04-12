import json
import os
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import random

def aggregrate_results(output_filenames, results_filename):
    results = {}
    for output_filename in output_filenames:
        outputs = {}
        if os.path.exists(output_filename):
            with open(output_filename, "r") as output_file:
                outputs = json.load(output_file)
        for arg_key, data in outputs.items():
            for subsample in [1, 5, 10, 15, 20]:
                subsample_arg_key = arg_key+f"_subsample_{subsample}"
                arithmetic_ngram_diversity = 0
                ancestral_ngram_diversity = 0
                greedy_ngram_diversity = 0
                arithmetic_vote_accuracy = 0
                ancestral_vote_accuracy = 0
                greedy_accuracy = 0
                random.shuffle(data)
                for example in data:
                    arithmetic_ngram_diversity += ngram_diversity(example["arithmetic_sampling"], subsample)
                    ancestral_ngram_diversity += ngram_diversity(example["ancestral_sampling"], subsample)
                    greedy_ngram_diversity += ngram_diversity(example["greedy_decoding"], subsample=1)
                    arithmetic_vote_accuracy += voting_accuracy(example["arithmetic_sampling"], example["gt"], subsample)
                    ancestral_vote_accuracy += voting_accuracy(example["ancestral_sampling"], example["gt"], subsample)
                    greedy_accuracy += numerical_accuracy(example["greedy_decoding"], example["gt"], subsample=1)
                if results.get(subsample_arg_key, None) is None:
                    results[subsample_arg_key] = {
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
                    results[subsample_arg_key]["ngram_diversity"]["arithmetic_sample"].append(arithmetic_ngram_diversity / len(data))
                    results[subsample_arg_key]["ngram_diversity"]["ancestral_sample"].append(ancestral_ngram_diversity / len(data))
                    results[subsample_arg_key]["ngram_diversity"]["greedy_decoding"].append(greedy_ngram_diversity / len(data))
                    results[subsample_arg_key]["accuracy"]["arithmetic_sample"].append(arithmetic_vote_accuracy / len(data))
                    results[subsample_arg_key]["accuracy"]["ancestral_sample"].append(ancestral_vote_accuracy / len(data))
                    results[subsample_arg_key]["accuracy"]["greedy_decoding"].append(greedy_accuracy / len(data))
    with open(results_filename, "w") as results_file:
        json.dump(results, results_file)

def analyze_results(output_filename, num_examples):
    outputs = {}
    if os.path.exists(output_filename):
        with open(output_filename, "r") as output_file:
            outputs = json.load(output_file)
    for arg_key, data in outputs.items():
        print(f"Results for {arg_key}:")
        arithmetic_ngram_diversity = 0
        ancestral_ngram_diversity = 0
        greedy_ngram_diversity = 0
        arithmetic_accuracy = 0
        ancestral_accuracy = 0
        arithmetic_vote_accuracy = 0
        ancestral_vote_accuracy = 0
        greedy_accuracy = 0
        for example in data:
            arithmetic_ngram_diversity += example["metrics"]["arithmetic_ngram_diversity"]
            ancestral_ngram_diversity += example["metrics"]["ancestral_ngram_diversity"]
            greedy_ngram_diversity += example["metrics"]["ancestral_ngram_diversity"]
            arithmetic_accuracy += example["metrics"]["arithmetic_accuracy"]
            ancestral_accuracy += example["metrics"]["ancestral_accuracy"]
            greedy_accuracy += example["metrics"]["greedy_accuracy"]
        for example in data: # FIXME
            arithmetic_vote_accuracy += voting_accuracy(example["arithmetic_sampling"], example["gt"])
            ancestral_vote_accuracy += voting_accuracy(example["ancestral_sampling"], example["gt"])
        print(f"Arithmetic N-gram Diversity: {arithmetic_ngram_diversity / num_examples:.4f}")
        print(f"Ancestral N-gram Diversity: {ancestral_ngram_diversity / num_examples:.4f}")
        print(f"Greedy N-gram Diversity: {greedy_ngram_diversity / num_examples:.4f}" )
        print(f"Arithmetic Macro-Average Accuracy: {arithmetic_accuracy / num_examples:.4f}")
        print(f"Ancestral Macro-Average Accuracy: {ancestral_accuracy / num_examples:.4f}")
        print(f"Arithmetic Majority-Voting Accuracy: {arithmetic_vote_accuracy / num_examples:.4f}")
        print(f"Ancestral Majority-Voting Accuracy: {ancestral_vote_accuracy / num_examples:.4f}")
        print(f"Greedy Decoding Accuracy: {greedy_accuracy / num_examples:.4f}")



    # plt.figure(figsize=(8, 6))
    # arg_key_accuracy = {}
    # for arg_key, data in outputs.items():
    #     arithmetic_vote_accuracy = 0
    #     ancestral_vote_accuracy = 0
    #     greedy_accuracy = 0
    #     for idx, example in enumerate(data):
    #         arithmetic_vote_accuracy += voting_accuracy(example["arithmetic_sampling"], example["gt"])
    #         ancestral_vote_accuracy += voting_accuracy(example["ancestral_sampling"], example["gt"])
    #         greedy_accuracy += example["metrics"]["greedy_accuracy"]
    #         if idx == 4:
    #             accuracy_5 = 0
    #         if idx == 9:
    #             accuracy_10 = 1
    #         if idx == 19:
    #             accuracy_20 = 2
    #     arg_key_accuracy[arg_key] = [accuracy_5, accuracy_10, accuracy_20]
    #     plt.plot(arg_key, arg_key_accuracy, label="Majority Voting Accuracy")
    # plt.xlabel("Strategy")
    # plt.ylabel("Metric Value")
    # plt.title("Performance Comparison")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("gemma_7b_it_comparison.png")

def plot_results(results_filename):
    with open(results_filename, "r") as results_file:
        results = json.load(results_file)
    sampled_sequences = [1, 5, 10, 15, 20]
    configs = {
        # 'num_examples=30, num_samples=20, temperature=0.7, shots=0, top_k=40, top_p=1.0': "T = 0.7, k = 40", 
        'num_examples=100, num_samples=20, temperature=0.5, shots=0, top_k=40, top_p=1.0': "T = 0.5, k = 40", 
        # 'num_examples=30, num_samples=20, temperature=0.3, shots=0, top_k=40, top_p=1.0': "T = 0.3, k = 40",
        # 'num_examples=30, num_samples=20, temperature=0.7, shots=0, top_k=20, top_p=1.0': "T = 0.7, k = 20", 
        # 'num_examples=30, num_samples=20, temperature=0.7, shots=0, top_k=50, top_p=1.0': "T = 0.7, no top k", 
        # 'num_examples=30, num_samples=20, temperature=1, shots=0, top_k=50, top_p=0.95':  "p = 0.95", 
        # 'num_examples=30, num_samples=20, temperature=1, shots=0, top_k=50, top_p=0.9':   "p = 0.90", 
        # "num_examples=200, num_samples=20, temperature=1, shots=0, top_k=50, top_p=0.95": "p = 0.95", 
        # "num_examples=200, num_samples=20, temperature=1, shots=0, top_k=50, top_p=0.9": "p = 0.9", 
    }
    for metric in ["accuracy", "ngram_diversity"]:
        for key in configs:
            for sampling in ["arithmetic_sample", "ancestral_sample"]:
                Y, Y_err = [], []
                for subsample in sampled_sequences:
                    subsample_arg_key = key + f"_subsample_{subsample}"
                    arr = results[subsample_arg_key][metric][sampling]
                    Y.append(np.mean(arr))
                    Y_err.append(np.std(arr))
                # plt.plot(sampled_sequences, Y, label = f"{sampling} {configs[key]}", marker='s')
                plt.errorbar(sampled_sequences, Y, yerr=Y_err, label = f"{sampling} {configs[key]}", marker='s')
        greedy_arg_key = f"num_examples=100, num_samples=20, temperature=0.5, shots=0, top_k=40, top_p=1.0_subsample_{subsample}"
        arr = results[greedy_arg_key][metric]["greedy_decoding"]
        Y = [np.mean(arr)]*len(sampled_sequences)
        Y_err = [np.std(arr)]*len(sampled_sequences)
        plt.errorbar(sampled_sequences, Y, yerr=Y_err, label = "greedy_decode", marker='o')
        # plt.plot(sampled_sequences, Y, label = "greedy_decode", marker='o')
        plt.legend()
        plt.xlabel("#Sampled Reasoning Paths")
        plt.ylabel(metric)
        plt.xticks(range(0, 25, 5))
        plt.title(f"{'Accuracy' if metric == 'accuracy' else 'Diversity'} Comparison (100 Q/A pairs)")
        plt.grid(True)
        plt.savefig(f"gemma_2b_it_{metric}.png")
        plt.clf()

if __name__ == "__main__":
    output_filenames = [f"gemma-2b-it__gsm8k__{seed}_output.json" for seed in [786, 275, 606, 402, 702]]
    results_filename = "gemma-2b-it__gsm8k__results.json"
    num_examples = 100
    # aggregrate_results(output_filenames, results_filename)
    plot_results(results_filename)
