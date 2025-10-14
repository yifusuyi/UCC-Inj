from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import json
import os
import sys
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name, input_filename, output_filename, thinking = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].lower() == 'true'

def split_think_content(text):
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
  model_name=model_name,
  tp=8,
  session_len=65535,
  max_batch_size=16,
  cache_max_entry_count=0.85,
  quant_policy=0,
  rope_scaling_factor=2.0,
  use_logn_attn=True,
  continuous_batching=True,
  enable_prefix_caching=True,
  max_prefill_token_num=10000,
)

gen_config = GenerationConfig(max_new_tokens=20000, do_sample=False)
pipe = pipeline(model_name, backend_config=backend_config)

finished_pid = set()
if os.path.exists(output_filename):
  with open(output_filename, "r", encoding='utf-8') as inf:
    for line in inf:
      try:
        data = json.loads(line)
        finished_pid.add(data['pid'])
      except:
        continue

with open(input_filename, "r", encoding="utf-8") as inf, \
     open(output_filename, "a", encoding="utf-8") as outf:
  
  prompts_to_process = []
  pids_to_process = []
  
  for line in inf:
    data = json.loads(line)
    pid = data['pid']
    if pid in finished_pid:
      continue
      
    user_content = ''' 
      Please think step by step and put your answer in \\boxed{}.
      Your answer must be an integer. For example, \\boxed{100}.
      You won't get score if the answer is not in the correct format, such as \\boxed{10,000}, \\boxed{1e4}, or \\boxed{$100$}.
      problem is：\n'''
    user_content += data['question']
    prompts_to_process.append(user_content)
    pids_to_process.append(pid)
  
  batch_size = backend_config.max_batch_size * 5
  for i in range(0, len(prompts_to_process), batch_size):
    batch_prompts = prompts_to_process[i:i+batch_size]
    batch_pids = pids_to_process[i:i+batch_size]
    
    try:
      results = pipe(batch_prompts, gen_config=gen_config)
      
      for j, result in enumerate(results):
        response = split_think_content(result.text)
        output_data = {
          'pid': batch_pids[j],
          'thinking_content': response['thinking_content'],
          'content': response['content']
        }
        outf.write(json.dumps(output_data, ensure_ascii=False) + "\n")
        outf.flush()
        
    except Exception as e:
      print(f"处理batch {i//batch_size}时出错: {str(e)}")
      continue