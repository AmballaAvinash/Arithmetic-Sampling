import torch
import torch.nn.functional as F
import numpy as np

# Constants
NEG_INF = torch.tensor(-1.0e7)
MIN_TEMPERATURE = torch.tensor(1e-4)

def _sequential_cumsum(arr, axis):
    """Sequential scan-based implementation of cumulative sum for PyTorch.

    Args:
        arr: PyTorch tensor to sum.
        axis: axis to sum over.

    Returns:
        PyTorch tensor of partial cumulative sums.
    """
    # Swap axes to bring the desired axis to the front
    arr_swapped = arr.transpose(0, axis)
    # Compute cumulative sum along the 0th axis
    cumsum = torch.cumsum(arr_swapped, dim=0)
    # Swap back the axes to their original order
    return cumsum.transpose(0, axis)




def _arithmetic_categorical(logits, codes):
    """Performs arithmetic sampling to select samples from a categorical distribution.

    Args:
        logits: tensor: [batch_size, vocab_size] float32 sequence of logits. (torch)
        codes: tensor: [batch_size] float32 codes for each batch element. (torch)

    Returns:
        A tuple (samples, new_codes) where `samples` are sampled indices with shape
        [batch_size], and `new_codes` are shape [batch_size] containing codes for
        the remaining suffix if doing ancestral sampling.
    """
    # We randomly permute the logits here at each timestep to avoid depending on
    # The default order of the vocabulary. This isn't strictly necessary.
    _, vocab_size = logits.shape
    perm = torch.randperm(logits.shape[1])
    invperm = torch.argsort(perm)

    logits = logits[:, perm]

    # Now we want to, for each element in the batch, get the normalized
    # probabilities, stack them in the unit interval into buckets, and figure
    # out what bucket the code falls into.
    probs = F.softmax(logits, dim=1)

    # Use the PyTorch cumsum to compute cumulative probabilities.
    cumprobs = _sequential_cumsum(probs, axis=1)

    # Ensure max value is at least 1.0 to prevent bucket width issues.
    max_probs = cumprobs.max(dim=1, keepdim=True)[0].expand_as(cumprobs)
    all_bucket_maxes = torch.where((cumprobs == max_probs) & (cumprobs < 1.0), 1.0, cumprobs)

    # Calculate code bucket mins and maxes.
    expanded_codes = codes.unsqueeze(1)
    bucket_maxes_lte_codes = all_bucket_maxes <= expanded_codes  #less than equal to 
    bucket_maxes_gt_codes = all_bucket_maxes > expanded_codes  # greater than
    code_bucket_mins = (all_bucket_maxes * bucket_maxes_lte_codes).max(dim=1)[0]
    code_bucket_maxes = ((all_bucket_maxes * bucket_maxes_gt_codes +
                          bucket_maxes_lte_codes.float() * 1.1).min(dim=1)[0])

    # Compute sampled indices.
    sampled_indices_permed = (
        (all_bucket_maxes * bucket_maxes_gt_codes +
         bucket_maxes_lte_codes.float() * 1.1).argmin(dim=1)
    )
    
    sampled_indices = torch.argmax(torch.nn.functional.one_hot(sampled_indices_permed, num_classes=vocab_size)[:, invperm], dim=1)

    
    # Compute new codes for remaining suffix.
    remainder_codes = (codes - code_bucket_mins) / (code_bucket_maxes - code_bucket_mins)

    return sampled_indices, remainder_codes




logits = torch.tensor([[3.9,4,6,2,-3],[4,5,6,8,9]])
codes = torch.tensor([0.45,0.988])

sampled_indices, remainder_codes = _arithmetic_categorical(logits,codes)

print(sampled_indices)
print(remainder_codes)