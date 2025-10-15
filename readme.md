- to test Luogu_Official_Contest dataset: exp1
- to test GSM8K dataset: exp3
- to apply character level tokenization: exp4
- to test models when prompted to denoise: exp5
- to calculate the similarity of hidden states between noied and clean input: exp6
- to calculate the attention weight to ASCII characters: exp7
- to test GSM8K_zh dataset: exp8
- to inject charcter `A` or `*` instead of VS characters: exp9
- to inject charcter visible ASCII characters instead of VS characters: exp10
- to randomize the injected character number: exp11
- to insert inrelevant prefix to problems: exp13
- to test different VS set size: exp14
- to give few-shots: exp15
- to categorize models' output to three types: exp18.

You should use `conda env create -f envir.yml` to load the environments. For experiments testing Llama models, you should have your `HF_token` in your environment varible.

You may read the `readme.md` in each experiments for details.

Note that you should have C++ compiler with standard C++23 to compile `.cc` files. You may need `GCC15`.