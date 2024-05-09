# from .utils  import GenerationMixin
import json
import os
from pathlib import Path
import random
import time
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams
from collections import Counter
from vllm import LLM, SamplingParams

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import sys
# import datasets
# from src.transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
# from src.transformers.models.auto.tokenization_auto import AutoTokenizer
def load_hf_data_set(split,dataset_name, dataset_subname):
        # breakpoint()
        data = {}
        # datasets.config.DOWNLOADED_DATASETS_PATH = Path('/work/pi_dhruvesh_pate_umass_edu/aparashar_umass_edu/datasets')
        data[split] = datasets.load_dataset(dataset_name,dataset_subname, split="validation" )
        return data[split]
def test():
    print("this is is a test")
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b").cuda()
    llm = LLM(model="google/gemma-2b",dtype="half")
    torch.device = 'cuda'
    

# Print the outputs.
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    # model.config.pad_token_id = model.config.eos_token_id
    # model.generation_config.pad_token_id = model.config.eos_token_id
    # data = random.sample(load_hf_data_set('validation','wmt14','de-en')['translation'],2)
    # # default_fwd_instruction = "Translate the following German sentence to an English sentence."
    # # default_fwd_input_prefix = "German sentence: "
    # # default_fwd_target_prefix = "English sentence: "
    # prompt_arr = [default_fwd_instruction,default_fwd_input_prefix]
    default_fwd_input_prefix = """Like each of the previous examples, answer the question with either "Yes" or "No".
    Provide reasoning for your answer.
    End your reasoning with the sentence: "So the answer is <your_answer>". Replace "<your_answer>" with your final answer, which should be either "Yes" or "No"."""
    turn_prefix_model = '<start_of_turn>model'
    turn_prefix_human = '<start_of_turn>human'
    turn_suffix = '<end_of_turn>'
    default_fwd_demos_prefix = "Here are some examples of reasoning based question answers."
    default_fwd_reasoning_prefix = "Reasoning: "
    default_fwd_question_prefix = "Question: "
    default_fwd_target_prefix = "Answer: "
    qa_exemplars = [
        {
            "question": "Do hamsters provide food for any animals?",
            "reasoning": "Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes.",
            "answer" : "Yes"
        },
        {
            "question": "Could Brooke Shields succeed at University of Pennsylvania?",
            "reasoning": "Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is yes.",
            "answer" : "Yes"
        },
        {
            "question": "Hydrogen’s atomic number squared exceeds number of Spice Girls?",
            "reasoning": "Hydroßen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is no.",
            "answer" : "No"
        },
        {
            "question": "Is it common to see frost during some college commencements?",
            "reasoning": "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is yes.",
            "answer" : "Yes"
        },
        {
            "question": "Could a llama birth twice during War in Vietnam (1945-46)?",
            "reasoning": "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no.",
            "answer" : "No"
        },
        {
            "question": "Would a pear sink in water?",
            "reasoning": "The density of a pear is about 0.6 g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no.",
            "answer" : "No"
        }
]

    
    demos = random.sample(qa_exemplars,6)
    demos = [[default_fwd_question_prefix + d["question"]]+ [default_fwd_reasoning_prefix + d["reasoning"] + default_fwd_target_prefix + d["answer"]] for d in demos]# def test(prompt_arr, model, tokenizer, data,demos, default_fwd_target_prefix, N = 10, temp = 0.5):

    
    output_dict = {}
    data =  [
        {
            "input": "Is it common to see frost during some college commencements?",
            "target_scores": {
                "Yes": 1,
                "No": 0
            },
            "target": "Yes. College commencement ceremonies often happen during the months of December, May, and sometimes June.  Frost isn't uncommon to see during the month of December, as it is the winter."
        },
        {
            "input": "Hydrogen's atomic number squared exceeds number of Spice Girls?",
            "target_scores": {
                "Yes": 0,
                "No": 1
            },
            "target": "No. Hydrogen is the first element and has an atomic number of one. To square a number, you multiply it by itself. The Spice Girls has five members."
        },
        {
            "input": "Would a pear sink in water?",
            "target_scores": {
                "Yes": 0,
                "No": 1
            },
            "target": "No. The density of a raw pear is about 0.59 g/cm^3. The density of water is about 1 g/cm^3. Objects only sink if they are denser than the surrounding fluid."
        },
        {
            "input": "Do hamsters provide food for any animals?",
            "target_scores": {
                "Yes": 1,
                "No": 0
            },
            "target": "Yes. Hamsters are prey animals. Prey animals provide food for predators. "
        }]
    for idx, d in enumerate(tqdm(data, desc="Predicting")):
        # prompt_arr.append(d['de'])
        # prompt_arr.append(default_fwd_target_prefix)
        # input_prompt = ('  ').join(prompt_arr)

        prompt_arr = [default_fwd_input_prefix] +[default_fwd_question_prefix + d['input']]
        prompt_arr = demos + [prompt_arr]
        prompt_arr = [element for sublist in prompt_arr for element in sublist]
        # print(prompt_arr)
        # input_prompt = turn_prefix_human + (' \n').join(prompt_arr) + turn_suffix + ' ' + turn_prefix_model
        input_prompt = (' \n').join(prompt_arr)
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95,n=20)
        outputs = llm.generate([input_prompt], sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        # breakpoint()
   
def calculate_bleu_and_ngram_diversity(reference, translations):
    bleu_score = corpus_bleu([reference]*len(translations), translations)

    n_values = [1, 2, 3, 4]
    total_unique_ngrams = 0
    ngram_diversity_score = 0
    for n in n_values:
        unique_ngrams = set()
        total_ngram_count = 0

        for translation in translations:
            # Compute n-grams
            # try:
            translation_ngrams = list(ngrams(translation.split(), n))
            # Count unique n-grams
            # breakpoint()

            total_ngram_count += len(list(translation_ngrams))
            unique_ngrams.update(translation_ngrams)
                # Count total n-grams
                
                # breakpoint()
            # except:
                # breakpoint()
                # continue
        # Update total counts
        try:
            total_unique_ngrams = len(unique_ngrams)
            ngram_diversity_score += total_unique_ngrams / total_ngram_count
        except:
            print(translation_ngrams)
            continue
        # breakpoint()
    return bleu_score, ngram_diversity_score

if __name__ == "__main__":
    test()
    
