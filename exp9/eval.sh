
./eval ./qwen3_30B_normal/A1.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl &&
./eval ./qwen3_30B_normal/A2.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl &&
./eval ./qwen3_30B_normal/A3.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl &&
./eval ./qwen3_30B_thinking/A1.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl &&
./eval ./qwen3_30B_thinking/A2.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl &&
./eval ./qwen3_30B_thinking/A3.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl


./eval ./qwen3_30B_normal/e1.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl &&
./eval ./qwen3_30B_normal/e2.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl &&
./eval ./qwen3_30B_normal/e3.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl &&
./eval ./qwen3_30B_thinking/e1.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl &&
./eval ./qwen3_30B_thinking/e2.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl &&
./eval ./qwen3_30B_thinking/e3.jsonl ../datasets/gsm8k/problemset_encoded0.jsonl result.jsonl
