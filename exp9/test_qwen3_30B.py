from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import json
import os
import sys
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


input_filename, output_filename, thinking = sys.argv[1], sys.argv[2], sys.argv[3],.lower() == 'true'

def split_think_content(text):
  """
  将包含<think>标签的文本分割为思考内容和正式内容
  返回格式: {"thinking_content": "思考部分文本", "content": "正式内容文本"}
  """
  pattern = r'<think>(.*?)</think>\s*(.*)'
  match = re.search(pattern, text, re.DOTALL)
  
  if match:
      return {
          "thinking_content": match.group(1).strip(),
          "content": match.group(2).strip()
      }
  else:
      return {
          "thinking_content": None,
          "content": text.strip()
      }

backend_config = TurbomindEngineConfig(
  model_name="qwen3-30b-a3b",
  tp=8,                        # 使用全部8张A800
  session_len=65535,            # 16K上下文长度
  max_batch_size=9,             # 保守批处理大小（确保精度）
  cache_max_entry_count=0.85,   # KV缓存比例
  quant_policy=0,               # 禁用量化（FP16精度）
  rope_scaling_factor=2.0,      # RoPE扩展因子
  use_logn_attn=True,           # 对数位置编码
  continuous_batching=True,     # 连续批处理
  enable_prefix_caching=True,   # 前缀缓存（优化长提示）
  max_prefill_token_num=20000,   # 最大预填充token数
)
gen_config = GenerationConfig(do_sample=False, max_new_tokens=20000)
pipe = pipeline('Qwen/Qwen3-30B-A3B',
                backend_config=backend_config)

def generate_llm_response(prompts):
  results = pipe(prompts, gen_config=gen_config)
  return results

finished_pid = []
pid_list = []
my_prompts = []

if os.path.exists(output_filename):
  with open(output_filename, "r", encoding ='utf-8') as inf:
    lines = inf.readlines()
    for line in lines:
      finished_pid.append(json.loads(line)['pid'])

with open(input_filename, "r", encoding="utf-8") as inf, \
     open(output_filename, "a", encoding="utf-8") as outf:
  lines = inf.readlines()
  for line in lines:
    data = json.loads(line)
    pid = data['pid']
    if pid in finished_pid:
      continue
    finished_pid.append(pid)
    pid_list.append(pid)
    user_content = ''' 
      我讲给你一个问题，你要回答该问题，问题的答案总是整数。
      你需要把你的答案放在 \\boxed{} 内。\\boxed{} 的内容是评判你答案正确性的唯一依据，它必须仅包含一个整数。
      例如，如果答案是 10000，你需要回答 \\boxed{10000}。回答 \\boxed{10,000}、\\boxed{1e4}、\\boxed{$10000$} 等都不得分。
      题目如下：\n'''
    user_content += data['question']
    if not thinking:
        user_content += '/nothink'
    my_prompts.append(user_content)
    
  results = pipe(my_prompts, gen_config=gen_config)
  pid_index = 0
  for i in results:
    response = split_think_content(i.text)
    result = {'pid': pid_list[pid_index], 'thinking_content' : response['thinking_content'], 'content': response['content']}
    pid_index += 1
    outf.write(json.dumps(result, ensure_ascii=False) + "\n")
