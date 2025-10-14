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
  model_name="qwen3-30b-a3b",
  tp=8,
  session_len=65535,
  max_batch_size=9,
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
    
    user_content = '''我将给你一个问题，你需要返回给我一份解决该问题的代码。
代码使用 C++ 写作。你可以给出文字分析过程，但是必须把代码用 ```\ncpp\n<codes>\n``` 的形式提供给我，即用 markdown 的代码框格式把代码框起来。
注意，如果你给出的代码没有正确被代码框格式包含，你将得到 0 分。
我的问题是：\n\n'''
    
    keyItems = ['background', 'description', 'inputFormat', 'outputFormat', 'hint']
    for i in keyItems:
      user_content += "## " + i.capitalize() + "\n\n" + data[i] + '\n\n'
      if i == 'outputFormat':
        samples = data['samples']
        user_content += '## samples\n\n'
        sample_cnt = 0
        for sample in samples:
          sample_cnt += 1
          user_content += '### InputSample' + str(sample_cnt) + '\n\n'
          user_content += '```\n' + sample[0] + '```\n'
          user_content += '### OutputSample' + str(sample_cnt) + '\n\n'
          user_content += '```\n' + sample[1] + '```\n'
      if not thinking:
        user_content += '/nothink'
    
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