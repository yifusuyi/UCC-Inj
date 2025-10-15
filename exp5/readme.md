Here is the experiment for test models when prompted to denoise.

## Run commands

You need to create subdirectories with corresponding names before running the following script.
(`normal` indicates to append `/nothing` in the end of prompt.)

```bash
bash test_qwen3_30B.sh
bash test_llama3.1_70B.sh
```

## Run evaluation

You need to provide **extra stack space** at runtime because `std::regex_search` is implemented by a recursive.

```bash
ulimit -s 1024000
g++ eval.cc -o eval -std=c++23 -O2
./eval jsonl_file_name ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl
```

Here is an example for evaluate the accurate of `qwen3_30B_normal/e0.jsonl`:

```bash
./eval qwen3_30B_normal/e0.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl
```

The accuracy will be appended as a new line in `result.jsonl`.