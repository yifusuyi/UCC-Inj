
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# 配置模型和文本
model_path = "Qwen/Qwen3-30B-A3B"
device = "cuda" if torch.cuda.is_available() else "cpu"
normal_text = "what is the result of 1 + 1?"
special_text = "what\u200B is\u200B the\u200B result\u200B of\u200B 1\u200B + 1?"  # 含零宽空格



tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def build_token_to_char_map(tokenizer):
    """创建可打印 ASCII 字符到 token ID 的映射"""
    token_to_char = {}
    
    # 遍历所有可打印 ASCII 字符 (32-126)
    for ascii_val in range(32, 127):
        char = chr(ascii_val)
        
        # 直接编码字符
        tokens = tokenizer.encode(char, add_special_tokens=False)
        #print(f"Character: {char}, Tokens: {tokens}, covrt_tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
        # 如果字符被拆分为单个 token
        if len(tokens) == 1:
            token_to_char[tokens[0]] = char
        # 如果字符被拆分为多个 token (如某些标点符号)
        elif tokens:
            # 使用第一个 token 作为主要映射
            token_to_char[tokens[0]] = char
            print(char) # 真有吗，貌似没有
    
    # 添加换行符等特殊字符
    special_chars = ["\n", "\t", "\r"]
    for char in special_chars:
        token_id = tokenizer.convert_tokens_to_ids(char)
        if token_id != tokenizer.unk_token_id:
            token_to_char[token_id] = char
#    print(token_to_char)
    return token_to_char

# 构建字符映射表
token_to_char = build_token_to_char_map(tokenizer)

# import sys; sys.exit()

# 加载模型

model = AutoModelForCausalLM.from_pretrained(
  model_path,
  trust_remote_code=True,
  attn_implementation="eager",
  torch_dtype=torch.bfloat16,
  device_map="auto"
).eval()

global mxdf
mxdf = 0

def process_text(text):

  chat_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(device)
  ids = chat_inputs["input_ids"][0]
#  print("The input ids: ", ids)
#  print("The input tokens: ", tokenizer.convert_ids_to_tokens(ids.tolist()))
  indices = [i for i, token in enumerate(ids.tolist()) if token in token_to_char]
  #indices2 = [0, 1, 2, len(chat_inputs[0]) - 2, len(chat_inputs[0]) - 1]
  indices2 = []
  indices += indices2
  #print(indices, indices2)
  if (len(ids) > 4010): 
    return [], [] # 会爆显存.jpg

  print(len(ids), ": tokens in the input")
  with torch.no_grad():
    outputs = model(**chat_inputs, output_attentions=True)
  
  # 获取所有注意力权重
  all_attentions = torch.stack([i.cpu() for i in outputs.attentions])
  # 转换为float32以支持NumPy
  all_attentions = all_attentions.float().numpy() # [num_layers, batch_size, num_heads, seq_len, seq_len]
  q_attn = all_attentions[:, 0, :, -1, :]
  ret_attn = [[sum(head[i] for i in indices) for head in layer] for layer in q_attn.tolist()]
  ret_attn2 = [[sum(head[i] for i in indices2) for head in layer] for layer in q_attn.tolist()]
  #global mxdf
  #mxdf = max(mxdf, max(max(abs(sum(head) - 1) for head in layer) for layer in q_attn.tolist()))
  return ret_attn, ret_attn2


datas = []
pid_list = []

small_text = "problemset_small.jsonl"

problem_prefix = '../datasets/gsm8k/problemset_encoded'
encoded_cnt_max = 3 #ovo
for encoded_cnt in range(1, encoded_cnt_max + 1):
  input_filename = f"{problem_prefix}{encoded_cnt}.jsonl"
  #input_filename = small_text #ovo
  with open(input_filename, "r", encoding="utf-8") as inf:
    lines = inf.readlines()
    datas.append(lines)
    for line in lines:
      data = json.loads(line)
      if encoded_cnt == encoded_cnt_max:
        pid_list.append(data['pid'])

ave_attns = [[0] * 32 for _ in range(48)]
ave_attns2 = [[0] * 32 for _ in range(48)]
num_attns = 0
with open('result_attn_source_30B.jsonl', 'a', encoding='utf-8') as outf:
  for pid in pid_list:
    user_content_prefix = '''
      我讲给你一个问题，你要回答该问题，问题的答案总是整数。
      这个问题的文本是被噪声污染过的，存在很多不可见的无意义字符。你要去掉这些无意义字符，然后理解文本，最后回答问题。
      你需要把你的答案放在 \\boxed{} 内。\\boxed{} 的内容是评判你答案正确性的唯一依据，它必须仅包含一个整数。
      例如，如果答案是 10000，你需要回答 \\boxed{10000}。回答 \\boxed{10,000}、\\boxed{1e4}、\\boxed{$10000$} 等都不得分。
      题目如下：\n
    '''
    user_content_prefix = ''
    for encoded_cnt in range(1, encoded_cnt_max + 1):
      user_content_special = user_content_prefix + json.loads(datas[encoded_cnt - 1][pid])['question']
      attns, attns2 = process_text(user_content_special)
      if (len(attns) == 0):
        continue
      ave_attns = [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(ave_attns, attns)]
      ave_attns2 = [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(ave_attns2, attns2)]
      num_attns += 1
      out = {}
      out['pid'] = pid
      out['encoded_cnt'] = encoded_cnt
      out['attention_in_ASCII'] = [[f'{cur:.5f}' for cur in row] for row in attns]
      #out['attention_in_spec'] = [[f'{cur:.5f}' for cur in row] for row in attns2]
      outf.write(json.dumps(out, ensure_ascii=False) + "\n")
      outf.flush()
      print("ok in pid =", pid, "encoded_cnt =", encoded_cnt)


ave_attns = [[a / num_attns for a in row] for row in ave_attns]
ave_attns2 = [[a / num_attns for a in row] for row in ave_attns2]
with open('result_attn_30B.jsonl', 'w', encoding='utf-8') as outf:
  out = {}
  out['average_attention_in_ASCII'] = [[f'{cur:.5f}' for cur in row] for row in ave_attns]
  #out['average_attention_in_spec'] = [[f'{cur:.5f}' for cur in row] for row in ave_attns2]
  outf.write(json.dumps(out, ensure_ascii=False) + "\n")
  outf.flush()

#print("Max diff:", mxdf) # 0.003 ?