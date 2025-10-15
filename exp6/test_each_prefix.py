import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import random
# 配置模型和文本
model_path = "/mnt/public/zay/Qwen3-30B-A3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

def insertcode(base: str, bytes_list: list[int]) -> str:
  encoded = [base]
  for byte in bytes_list:
    if byte < 16:
      codepoint = 0xFE00 + byte
    else:
      codepoint = 0xE0100 + (byte - 16)
    encoded.append(chr(codepoint))
  return ''.join(encoded)


def insert_random_code(text : str, min_bytes : int, max_bytes : int) -> str:
  encoded = []
  for char in text:
    num_bytes = random.randint(min_bytes, max_bytes)
    random_bytes = [random.randint(0, 255) for _ in range(num_bytes)]
    encoded_char = insertcode(char, random_bytes)
    encoded.append(encoded_char)
  return ''.join(encoded)

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
  model_path,
  trust_remote_code=True,
  output_hidden_states=True,
  torch_dtype=torch.bfloat16
).to(device).eval()

def process_text(text):
  chat = [{"role": "user", "content": text}]
  chat_inputs = tokenizer.apply_chat_template(
    chat, 
    tokenize=True, 
    add_generation_prompt=False,
    return_tensors="pt"
  ).to(device)
  
  with torch.no_grad():
    outputs = model(chat_inputs)
  
  # 获取所有层的隐藏状态 [layers, batch_size, seq_len, hidden_dim]
  all_hidden = torch.stack(outputs.hidden_states)
  # 提取最后一个token的所有层隐藏状态 [layers, hidden_dim]
  last_token_hidden = all_hidden[:, 0, -1, :]
  
  # 转换为float32以支持NumPy
  last_token_hidden = last_token_hidden.float().cpu().numpy()

  return last_token_hidden

def calc_similarity(normal_hidden, special_hidden):
# 逐层计算余弦相似度
  num_layers = normal_hidden.shape[0]
  similarities = []
  for layer_idx in range(num_layers):
    vec_normal = normal_hidden[layer_idx].reshape(1, -1)
    vec_special = special_hidden[layer_idx].reshape(1, -1)
    sim = cosine_similarity(vec_normal, vec_special)[0][0]
    similarities.append(sim)
  return similarities

problem_prefix = '../datasets/gsm8k/problemset_encoded'
encoded_cnt_max = 3

datas = []
pid_list = []

for encoded_cnt in range(0, encoded_cnt_max + 1):
  input_filename = f"{problem_prefix}{encoded_cnt}.jsonl"
  with open(input_filename, "r", encoding="utf-8") as inf:
    lines = inf.readlines()
    datas.append(lines)
    for line in lines:
      data = json.loads(line)
      if encoded_cnt == 0:
        pid_list.append(data['pid'])
      
with open('result_each_prefix.jsonl', 'a', encoding='utf-8') as outf:
  for pid in pid_list:
    if pid >= 10:
      break
    data_0 : str = json.loads(datas[0][pid - 1])
    data_0_splited_by_spaces = data_0['question'].split(' ')
    prefix = ''
    prefix_len = 0 #count by words
    for i in data_0_splited_by_spaces:
      prefix_len += 1
      prefix += i
      input_list = [prefix]
      for encoded_cnt in range(1, encoded_cnt_max + 1):
        input_list.append(insert_random_code(prefix, encoded_cnt, encoded_cnt))
      normal_hidden = process_text(input_list[0])
      for encoded_cnt in range(1, encoded_cnt_max + 1):
        user_content_special = input_list[encoded_cnt]
        special_hidden = process_text(user_content_special)
        similarities = calc_similarity(normal_hidden, special_hidden)
        out = {}
        out['pid'] = pid
        out['prefix_len'] = prefix_len
        out['encoded_cnt'] = encoded_cnt
        out['similarities'] = [f'{sim:.5f}' for sim in similarities]
        outf.write(json.dumps(out, ensure_ascii=False) + "\n")
        outf.flush()
