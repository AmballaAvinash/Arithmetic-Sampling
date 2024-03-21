from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import decoding

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)


# Greedy search
# encode context the generation is conditioned on
batch_size = 2
num_decodes = 2

input_prompt = ['I enjoy walking with my cute dog', 'Today is a beautiful day, and']
input_ids = tokenizer(input_prompt, return_tensors='pt').to(torch_device).input_ids
input_ids = input_ids.repeat(1,num_decodes).reshape(num_decodes*batch_size,-1)


# Arithemtic sampling
rng = torch.Generator().manual_seed(42)  # Set a random seed for reproducibility
codes = decoding._make_default_codes(batch_size, num_decodes, rng)


max_len = 20
while True:
    
    outputs = model(input_ids)
    next_token_logits = outputs.logits[:, -1, :]
    
    next_tokens, codes = decoding._arithmetic_categorical(next_token_logits, num_decodes, codes)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    max_len = max_len-1
    if max_len == 0:
        break


print("Output:\n" + 100 * '-')
for i in range(input_ids.shape[0]):
    print("new output")
    print(tokenizer.decode(input_ids[i], skip_special_tokens=True))
