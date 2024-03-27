# Begin experiments with Flan-T5
python3 src/reasoning/main.py --num-examples 30 --num-samples 10 --temperature 0.5 --num-shots 0
python3 src/reasoning/main.py --num-examples 30 --num-samples 10 --temperature 0.5 --num-shots 1
python3 src/reasoning/main.py --num-examples 30 --num-samples 10 --temperature 0.5 --num-shots 4
python3 src/reasoning/main.py --num-examples 30 --num-samples 10 --temperature 1.0 --num-shots 0
python3 src/reasoning/main.py --num-examples 30 --num-samples 10 --temperature 1.0 --num-shots 1
python3 src/reasoning/main.py --num-examples 30 --num-samples 10 --temperature 1.0 --num-shots 4
# End experiments with Flan-T5

# Begin experiments with Gemma-2b-it
# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 0.5 --num-shots 0 --model gemma-2b-it
# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 0.5 --num-shots 1 --model gemma-2b-it
# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 0.5 --num-shots 4 --model gemma-2b-it
# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 1.0 --num-shots 0 --model gemma-2b-it
# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 1.0 --num-shots 1 --model gemma-2b-it
# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 1.0 --num-shots 4 --model gemma-2b-it
# End experiments with Gemma-2b-it

# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 0.5 --num-shots 0 --model gemma-7b-it
# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 0.5 --num-shots 1 --model gemma-7b-it
# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 0.5 --num-shots 4 --model gemma-7b-it
# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 1.0 --num-shots 0 --model gemma-7b-it
# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 1.0 --num-shots 1 --model gemma-7b-it
# python3 src/reasoning/main.py --num-examples 10 --num-samples 10 --temperature 1.0 --num-shots 4 --model gemma-7b-it