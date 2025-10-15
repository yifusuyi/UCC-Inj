Here is the experiment to randomize the injected character number.

## Create Dataset

```bash
cd dataset
g++ encoder_random_inserted_num.cc -o encoder_random_inserted_num -O2 -std=c++23
./encoder_random_inserted_num
```

## Run Commands

You need to create subdirectories with corresponding names before running the following script.

```bash
bash test_qwen3_30B.sh
```

## Run Evaluation

You need to provide **extra stack space** at runtime because `std::regex_search` is implemented by a recursive.

```bash
ulimit -s 1024000
g++ eval.cc -o eval -std=c++23 -O2
eval.sh
```

The accuracy will be appended as new lines in `result.jsonl`.