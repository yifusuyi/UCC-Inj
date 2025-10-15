python test_qwen3_30B.py ./dataset/problemset_encodedA1.jsonl qwen3_30B_normal/A1.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encodedA2.jsonl qwen3_30B_normal/A2.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encodedA3.jsonl qwen3_30B_normal/A3.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encodedA1.jsonl qwen3_30B_thinking/A1.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encodedA2.jsonl qwen3_30B_thinking/A2.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encodedA3.jsonl qwen3_30B_thinking/A3.jsonl True &&

python test_qwen3_30B.py ./dataset/problemset_encoded1.jsonl qwen3_30B_normal/e1.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded2.jsonl qwen3_30B_normal/e2.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded3.jsonl qwen3_30B_normal/e3.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded1.jsonl qwen3_30B_thinking/e1.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encoded2.jsonl qwen3_30B_thinking/e2.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encoded3.jsonl qwen3_30B_thinking/e3.jsonl True