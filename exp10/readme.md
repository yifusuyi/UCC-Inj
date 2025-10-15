Here is the experiment to inject charcter visible ASCII characters instead of VS characters.

## Create Dataset

```bash
cd dataset
g++ encoder_normal_ASCII.cc -o encoder_normal_ASCII -O2 -std=c++23
./encoder_A
g++ encoder_visible_ASCII.cc -o encoder_visible_ASCII -O2 -std=c++23
./encoder_star
```

## Run Commands

You need to create subdirectories with corresponding names before running the following script.

```bash
bash test.sh
```

## Run Evaluation

You need to provide **extra stack space** at runtime because `std::regex_search` is implemented by a recursive.

```bash
ulimit -s 1024000
g++ eval.cc -o eval -std=c++23 -O2
eval.sh
```

The accuracy will be appended as new lines in `result.jsonl`.