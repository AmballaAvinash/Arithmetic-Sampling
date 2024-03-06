import os
import time
from modeling import ArithmeticSampler
from configuration_utils import ArithmeticSamplingGenerationConfig
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

start_time = time.time()
prompt = "The slow"
checkpoint = "distilbert/distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt")

generation_config = ArithmeticSamplingGenerationConfig(
    arithmetic_sampling=True,
    do_sample=True, 
    max_new_tokens=50
)

model = ArithmeticSampler(BartForConditionalGeneration).from_pretrained(checkpoint)
outputs = model.generate(**inputs, generation_config=generation_config)
generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(generated[0])
end_time = time.time()
print("Time taken: ", end_time - start_time, " seconds")