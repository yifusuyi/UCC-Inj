python test_qwen3_30B.py  ../datasets/gsm8k/problemset_encoded1.jsonl qwen3_30B_normal/e1.jsonl False &&
python test_qwen3_30B.py  ../datasets/gsm8k/problemset_encoded2.jsonl qwen3_30B_normal/e2.jsonl False &&
python test_qwen3_30B.py  ../datasets/gsm8k/problemset_encoded3.jsonl qwen3_30B_normal/e3.jsonl False &&
python test_qwen3_30B.py  ../datasets/gsm8k/problemset_encoded1.jsonl qwen3_30B_thinking/e1.jsonl True &&
python test_qwen3_30B.py  ../datasets/gsm8k/problemset_encoded2.jsonl qwen3_30B_thinking/e2.jsonl True &&
python test_qwen3_30B.py  ../datasets/gsm8k/problemset_encoded3.jsonl qwen3_30B_thinking/e3.jsonl True