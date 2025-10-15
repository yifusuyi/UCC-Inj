Here is the experiment to insert inrelevant prefix to problems.

## Create Dataset

```bash
cd dataset
g++ encoder_addprefix.cc -o encoder_addprefix -O2 -std=c++23
./encoder_addprefix
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
bash eval.sh
```

The accuracy will be appended as new lines in `result.jsonl`.