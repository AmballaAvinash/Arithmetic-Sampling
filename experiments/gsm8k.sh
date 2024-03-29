# Begin experiments with Flan-T5
python3 src/reasoning/main.py --num-examples 100 > flan_t5_large-gsm8k-eval.txt
# End experiments with Flan-T5

# Begin experiments with Gemma-2b-it
python3 src/reasoning/main.py --num-examples 100 --model gemma-2b-it > gemma_2b_it-gsm8k-eval.txt
# End experiments with Gemma-2b-it

# Begin experiments with Gemma-7b-it
python3 src/reasoning/main.py --num-examples 100 --model gemma-7b-it > gemma_7b_it-gsm8k-eval.txt
# End experiments with Gemma-7b-it