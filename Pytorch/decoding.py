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




def _arithmetic_categorical(logits, num_decodes, codes):  # decodes next index given logits and previous code point.  
    """Performs arithmetic sampling to select samples from a categorical distribution.

    Args:
        logits: tensor: [batch_size*num_decodes, vocab_size] float32 sequence of logits. (torch)
        codes: tensor: [batch_size*num_decodes] float32 codes for each batch element. (torch)

    Returns:
        A tuple (samples, new_codes) where `samples` are sampled indices with shape
        [batch_size], and `new_codes` are shape [batch_size] containing codes for
        the remaining suffix if doing ancestral sampling.
    """
    # We randomly permute the logits here at each timestep to avoid depending on
    # The default order of the vocabulary. This isn't strictly necessary.

    _, vocab_size = logits.shape

    # print("vocab size {}".format(vocab_size))
    perm = torch.randperm(logits.shape[1])
    invperm = torch.argsort(perm)  # recovering the original perumuation

    logits = logits[:, perm]
    
    # print("permutation {}".format(perm))
    # print("inverse permutation {}".format(invperm))
    # print("permuted logits {}".format(logits))

    # Now we want to, for each element in the batch, get the normalized
    # probabilities, stack them in the unit interval into buckets, and figure
    # out what bucket the code falls into.
    probs = F.softmax(logits, dim=1)
    
    # print("softmax over logits {}".format(probs))

    # Use the PyTorch cumsum to compute cumulative probabilities.
    cumprobs = _sequential_cumsum(probs, axis=1)  # CDF 
    
    # print("cumsum over probs {}".format(cumprobs))
    

    # Ensure max value is at least 1.0 to prevent bucket width issues.
    max_probs = cumprobs.max(dim=1, keepdim=True)[0].expand_as(cumprobs)
    all_bucket_maxes = torch.where((cumprobs == max_probs) & (cumprobs < 1.0), 1.0, cumprobs)  # all_bucket_maxes = cumprobs with last index to 1 (since CDF over all vocab space = 1)

    # Calculate code bucket mins and maxes.
    expanded_codes = codes.unsqueeze(1)
    bucket_maxes_lte_codes = all_bucket_maxes <= expanded_codes  #True if cumsum is less than equal to code point
    bucket_maxes_gt_codes = all_bucket_maxes > expanded_codes  # #True if cumsum is greater than equal to code point. So bucket_maxes_lte_codes and bucket_maxes_gt_codes are disjoint
    code_bucket_mins = (all_bucket_maxes * bucket_maxes_lte_codes).max(dim=1)[0]  # maximum cumsum that is less than equal to code point. 
    code_bucket_maxes = ((all_bucket_maxes * bucket_maxes_gt_codes +
                          bucket_maxes_lte_codes.float() * 1.1).min(dim=1)[0])  # minimum cumsum that is greater than equal to code point. 

    # Compute sampled indices.
    sampled_indices_permed = (
        (all_bucket_maxes * bucket_maxes_gt_codes +
         bucket_maxes_lte_codes.float() * 1.1).argmin(dim=1)  # minimum cumsum index that is greater than equal to code point. 
    )
    
    # print("max_probs {}".format(max_probs))
    # print("all_bucket_maxes {}".format(all_bucket_maxes))
    # print("expanded_codes {}".format(expanded_codes))
    # print("bucket_maxes_lte_codes {}".format(bucket_maxes_lte_codes))
    # print("bucket_maxes_gt_codes {}".format(bucket_maxes_gt_codes))
    # print("code_bucket_mins {}".format(code_bucket_mins))
    
    # print("code_bucket_maxes {}".format(code_bucket_maxes))
    # print("sampled_indices_permed {}".format(sampled_indices_permed))
    
    
    sampled_indices = torch.argmax(torch.nn.functional.one_hot(sampled_indices_permed, num_classes=vocab_size)[:, invperm], dim=1)   # making pertuted sampled indices to original order sample indices

    
    # Compute new codes for remaining suffix.
    remainder_codes = (codes - code_bucket_mins) / (code_bucket_maxes - code_bucket_mins)

    return sampled_indices, remainder_codes




# logits = torch.tensor([[3.9,4,6,2,-3],[4,5,6,8,9]])
# codes = torch.tensor([0.45,0.988])
# num_decodes = 2
# sampled_indices, remainder_codes = _arithmetic_categorical(logits,num_decodes, codes)

# print("sampled indices {}".format(sampled_indices))
# print("remainder_codes {}".format(remainder_codes))




def _make_default_codes(batch_size, num_decodes,
                        rng):
  """Make default codebook for a batch of `num_decodes` samples.

  The codes are initialized evenly spaced in the unit interval, with a random
  offset applied. This lets them evenly cover the sample space while also
  providing an unbiased estimate of any sample average.

  Args:
    batch_size: size of input batch.
    num_decodes: number of samples per batch element.
    rng: random seed.

  Returns:
    [batch_size* num_decodes] array of codes.
  """
  offset = torch.rand(batch_size, 1) # batch size is for generating parallel sequences in batch 
  codes = torch.tile(
        torch.unsqueeze(
            torch.arange(1, num_decodes + 1, dtype=torch.float32) / (num_decodes + 1),  # codes spaces at equal interval from [0,1] for diverse gereration
            dim=0), (batch_size, 1))
  return torch.flatten(torch.fmod(codes + offset, 1.0)) # offset is removing bias and fmod is making codes to [0,1]


# batch_size = 2
# num_decodes = 2
# rng = torch.Generator().manual_seed(42)  # Set a random seed for reproducibility
# result = _make_default_codes(batch_size, num_decodes, rng)
# print(result)




def arithmetic_code(word, char_freq):
    freq = list(char_freq.values())
    prob = {}
    for char in char_freq.keys():
        prob[char] = char_freq[char]/np.sum(freq)
    
    # ordering of charecters doesn't matter 
    lb = {}  # lower bound
    ub = {}
    
    start = 0
    for char in char_freq.keys():
        lb[char] = start
        ub[char] = start+prob[char]
        
        start = start+prob[char]
    
    
    
    code_lb = [lb[word[0]]]
    code_ub = [ub[word[0]]]
    
    for i in range(1,len(word)):
        
        code_lb.append(code_lb[i-1] + lb[word[i]]*(code_ub[i-1]-code_lb[i-1]))  # c_i  = m + lb[i]* (M-m) 
        code_ub.append(code_lb[i-1] + ub[word[i]]*(code_ub[i-1]-code_lb[i-1]))
        
        
    return code_lb, code_ub  
  
    
# word = "HELLO"
# char_freq = {"H":1, "E":1, "L":2, "O":1}
# code_lb, code_ub =  arithmetic_code(word, char_freq)
    
# print(code_lb)
# print(code_ub)

# print("probability is {}".format(np.array(code_ub)-np.array(code_lb)))
