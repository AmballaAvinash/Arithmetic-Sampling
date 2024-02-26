# from .utils  import GenerationMixin
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
def test():
    print("this is is a test")
    


    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id

    input_prompt = "Today is a beautiful day, and"
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()

    # instantiate logits processors
    logits_processor = LogitsProcessorList(
        [
            MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ]
    )
    # instantiate logits processors
    logits_warper = LogitsProcessorList(
        [
            TopKLogitsWarper(50),
            TemperatureLogitsWarper(0.7),
        ]
    )

    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    torch.manual_seed(0)
    outputs = model.arithmetic_sample(
        input_ids,
        logits_processor=logits_processor,
        logits_warper=logits_warper,
        stopping_criteria=stopping_criteria,
    )

    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
if __name__ == "__main__":
    test()