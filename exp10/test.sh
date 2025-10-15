

python test_qwen3_30B.py ./dataset/problemset_encoded1.jsonl qwen3_30B_normal/e1.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded2.jsonl qwen3_30B_normal/e2.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded3.jsonl qwen3_30B_normal/e3.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded1.jsonl qwen3_30B_thinking/e1.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encoded2.jsonl qwen3_30B_thinking/e2.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encoded3.jsonl qwen3_30B_thinking/e3.jsonl True &&

python test_qwen3_30B.py ./dataset/problemset_normal_ASCII1.jsonl qwen3_30B_normal/n1.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_normal_ASCII2.jsonl qwen3_30B_normal/n2.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_normal_ASCII3.jsonl qwen3_30B_normal/n3.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_normal_ASCII1.jsonl qwen3_30B_thinking/n1.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_normal_ASCII2.jsonl qwen3_30B_thinking/n2.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_normal_ASCII3.jsonl qwen3_30B_thinking/n3.jsonl True 
