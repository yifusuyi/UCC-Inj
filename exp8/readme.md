Here is the experiment for testing model on the Chinses version of GSM8K.

## Run commands


You need to create subdirectories with corresponding names before running the following script.

```bash
bash test.sh
```

## Run evaluation

You need to provide **extra stack space** at runtime because `std::regex_search` is implemented by a recursive.

```bash
ulimit -s 1024000
g++ eval.cc -o eval -std=c++23 -O2
eval.sh
```

The accuracy will be appended as new lines in `result.jsonl`.