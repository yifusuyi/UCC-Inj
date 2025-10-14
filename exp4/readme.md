Here is the experiment for test models with character level tokenization.

## Run commands

You need to create subdirectories with corresponding names before running the following script.

```bash
bash test_qwen.sh
bash test_llama.sh
bash test_A.sh #This is the experiment to apply character level tokenization to 1-Inj with character 'A'
```

## Run evaluation

You need to provide **extra stack space** at runtime because `std::regex_search` is implemented by a recursive.

```bash
ulimit -s 1024000
g++ eval.cc -o eval -std=c++23 -O2
./eval jsonl_file_name ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl
```

Here is an example for evaluate the accurate of `Qwen3-32B/e0.jsonl`:

```bash
./eval Qwen3-32B /e0.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl
```

The accuracy will be appended as a new line in `result.jsonl`.