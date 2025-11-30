import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('test-00000-of-00001.parquet')

# 转换为 JSONL
df.to_json('../test.jsonl', orient='records', lines=True)