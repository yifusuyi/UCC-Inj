## Run commands

```bash
bash test_qwen3.sh
bash test_qwen2.5.sh
bash test_llama.sh
bash test_deepseek_distilled_qwen.sh
```

## Run evaluation

You need to provide **extra stack space** at runtime because `std::regex_search` is implemented by a recursive.

```bash
ulimit -s 1024000
g++ eval.cc -o eval -std=c++23 -O2
./eval jsonl_file_name ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl
```

Here is an example for evaluate the accurate of `qwen3_32B_normal/e0.jsonl`:

```bash
./eval qwen3_32B_normal/e0.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl
```

The accuracy will be appended as a new line in `result.jsonl`.