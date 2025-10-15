
python test_qwen3_30B.py ./dataset/problemset_encoded_0_2.jsonl qwen3_30B_thinking/e02.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encoded_0_1.jsonl qwen3_30B_thinking/e01.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encoded_1_2.jsonl qwen3_30B_thinking/e12.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encoded_1_3.jsonl qwen3_30B_thinking/e13.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encoded_1_4.jsonl qwen3_30B_thinking/e14.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encoded_2_3.jsonl qwen3_30B_thinking/e23.jsonl True &&
python test_qwen3_30B.py ./dataset/problemset_encoded_2_4.jsonl qwen3_30B_thinking/e24.jsonl True &&


python test_qwen3_30B.py ./dataset/problemset_encoded_0_2.jsonl qwen3_30B_normal/e02.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded_0_1.jsonl qwen3_30B_normal/e01.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded_1_2.jsonl qwen3_30B_normal/e12.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded_1_3.jsonl qwen3_30B_normal/e13.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded_1_4.jsonl qwen3_30B_normal/e14.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded_2_3.jsonl qwen3_30B_normal/e23.jsonl False &&
python test_qwen3_30B.py ./dataset/problemset_encoded_2_4.jsonl qwen3_30B_normal/e24.jsonl False
