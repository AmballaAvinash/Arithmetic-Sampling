import random
import torch
from datasets import load_dataset, Dataset
from nltk.util import ngrams

GSM8K_EXEMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
        "answer": "6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        "answer": "5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
        "answer": "39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "reasoning": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
        "answer": "8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "reasoning": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
        "answer": "9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "reasoning": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
        "answer": "29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "reasoning": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
        "answer": "33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "reasoning": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
        "answer": "8",
    },
]


MODEL_ENUM = {
    "flan-t5-base": ("google", "flan-t5-base"),
    "flan-t5-large": ("google", "flan-t5-large"),
    "gemma-2b-it": ("google", "gemma-2b-it"),
    "gemma-7b-it": ("google", "gemma-7b-it"),
    "llama-2-7b": ("meta-llama", "Llama-2-7b"),
}

DATASET_ENUM = {"gsm8k": ("gsm8k", "main", "test", GSM8K_EXEMPLARS)}

ANSWER_TRIGGER = "Therefore, the answer (number) is"

COT_TRIGGER = "Let us think step by step using mathematical reasoning."


def load_HF_dataset(
    dataset_name: str, dataset_subname: str, dataset_split: str
) -> Dataset:
    """Load a dataset from the Hugging Face Hub.

    Args:
        dataset_name (str): Name of the dataset to load.
        dataset_subname (str): Name of the sub-dataset to load.
        dataset_split (str): Name of the split to load.

    Returns:
        Dataset: The dataset loaded from the Hugging Face Hub.
    """
    return load_dataset(
        dataset_name, dataset_subname, split=dataset_split, trust_remote_code=True
    )


def generate_few_shot_exemplars(dataset_name: str, num_examples: int = -1) -> str:
    """Generate a few-shot learning prompt from a dataset.

    Args:
        dataset_name (str): Name of the dataset to load exemplars for.
        num_examples (int, optional): The number of exemplars to sample. Defaults to -1 (all exemplars).

    Returns:
        str: A few-shot learning prompt.
    """

    # Load the exemplars for the dataset
    exemplars = DATASET_ENUM[dataset_name][3]

    # Randomly shuffle the exemplars
    exemplar_indices = list(range(len(exemplars)))
    random.shuffle(exemplar_indices)

    # If `num_examples` is not -1, choose the first `num_examples` exemplars
    if num_examples != -1:
        exemplar_indices = exemplar_indices[:num_examples]

    # Choose the first `num_examples` exemplars
    prompt_prefix = ""
    for index in exemplar_indices:
        exemplar = exemplars[index]
        question, reasoning, answer = (
            exemplar["question"],
            exemplar["reasoning"],
            exemplar["answer"],
        )
        prompt_prefix += f"Question: {question} \nAnswer: {COT_TRIGGER} {reasoning} {ANSWER_TRIGGER} {answer}.\n\n"

    return prompt_prefix


def ngram_diversity(sequences: list[str]) -> float:
    n_values = [1, 2, 3, 4]
    total_unique_ngrams = 0
    ngram_diversity_score = 0
    for n in n_values:
        unique_ngrams = set()
        total_ngram_count = 0
        for sequence in sequences:
            sequence_ngrams = list(ngrams(sequence.split(), n))
            total_ngram_count += len(list(sequence_ngrams))
            unique_ngrams.update(sequence_ngrams)
        total_unique_ngrams = len(unique_ngrams)
        ngram_diversity_score += total_unique_ngrams / (
            total_ngram_count + torch.finfo(torch.float32).eps
        )
    return ngram_diversity_score


def numerical_accuracy(sequences: list[str], ground_truth: str) -> float:
    return sum(
        [
            int(sequence.split(COT_TRIGGER)[-1].strip("\n") != ground_truth)
            for sequence in sequences
        ]
    ) / len(sequences)
