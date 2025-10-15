Here is the experiment to categorize models' output to three types.

## Run Commands

Firstly, you should copy the output model generated to correspond subdirectories. And create another subdirectory named `group`.

For example, you have `qwen3_30B_normal/e1.jsonl`

```bash
mkdir -p groups
mkdir -p groups/qwen3_30B_normal
python test.py qwen3_30B_normal e1 0,1,2,3,4,5,6,7
# the last parameter is the indexes of GPU you use.
```

## Run Evaluation

You need to provide **extra stack space** at runtime because `std::regex_search` is implemented by a recursive.

```bash
ulimit -s 1024000
g++ eval.cc -o eval -std=c++23 -O2
g++ eval2.cc -o eval2 -std=c++23 -O2

bash eval.sh
```

Run `./eval qwen3_30B_normal/e1.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl` to calculate the proportion and accuracy of each type in `qwen3_30B_normal/e1.jsonl`.

Run `./eval qwen3_30B_normal/e1.jsonl qwen3_30B_normal_prompted/e1.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl` to calculate the number of answers that is grouped to A when prompted to denoise but grouped to C when not prompted. Ofcourse you should run `test.py` for `qwen3_30B_normal_prompted/e1.jsonl` first.