from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import json
import os
import sys
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

input_filename, output_filename, thinking = sys.argv[1], sys.argv[2], sys.argv[3].lower() == 'true'

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
  model_name="llama3.1-80B",
  tp=8,                        # 使用全部8张A800
  session_len=30000,            # 16K上下文长度
  max_batch_size=9,             # 保守批处理大小（确保精度）
  cache_max_entry_count=0.85,   # KV缓存比例
  quant_policy=0,               # 禁用量化（FP16精度）
  rope_scaling_factor=2.0,      # RoPE扩展因子
  use_logn_attn=True,           # 对数位置编码
  continuous_batching=True,     # 连续批处理
  enable_prefix_caching=True,   # 前缀缓存（优化长提示）
  max_prefill_token_num=10000,   # 最大预填充token数
)
gen_config = GenerationConfig(do_sample=False, max_new_tokens=20000)
pipe = pipeline('meta-llama/Llama-3.1-70B-Instruct',
                backend_config=backend_config)

def generate_llm_response(prompts):
  results = pipe(prompts, gen_config=gen_config)
  return results

finished_pid = []
pid_list = []
my_prompts = []

# 读取已完成的pid
finished_pid = set()
if os.path.exists(output_filename):
    with open(output_filename, "r", encoding='utf-8') as inf:
        for line in inf:
            try:
                data = json.loads(line)
                finished_pid.add(data['pid'])
            except:
                continue

# 处理未完成的数据
with open(input_filename, "r", encoding="utf-8") as inf, \
     open(output_filename, "a", encoding="utf-8") as outf:
    
    # 收集所有需要处理的数据
    prompts_to_process = []
    pids_to_process = []
    
    for line in inf:
        data = json.loads(line)
        pid = data['pid']
        if pid in finished_pid:
            continue
            
        user_content = ''' 
        I will ask you a problem, and you must answer it. The answer to this problem is always an integer.  
      The text of this problem has been corrupted by noise, containing many invisible and meaningless characters. You need to remove these meaningless characters, then understand the text, and finally answer the question.  

      You must place your answer within \\boxed{}. The content of \\boxed{} is the sole basis for evaluating the correctness of your answer, and it must contain only a single integer.  
      For example, if the answer is 10000, you should respond with \\boxed{10000}. Answers such as \\boxed{10,000}, \\boxed{1e4}, or \\boxed{$10000$} will not be scored.
      The problem is：\n'''
        user_content += data['question']
        # if not thinking:
        #     user_content += '/nothink'
            
        prompts_to_process.append(user_content)
        pids_to_process.append(pid)
    
    # 分批处理数据
    batch_size = backend_config.max_batch_size * 5
    for i in range(0, len(prompts_to_process), batch_size):
        batch_prompts = prompts_to_process[i:i+batch_size]
        batch_pids = pids_to_process[i:i+batch_size]
        
        try:
            # 处理当前batch
            results = pipe(batch_prompts, gen_config=gen_config)
            
            # 写入当前batch的结果
            for j, result in enumerate(results):
                response = split_think_content(result.text)
                output_data = {
                    'pid': batch_pids[j],
                    'thinking_content': response['thinking_content'],
                    'content': response['content']
                }
                outf.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                outf.flush()  # 确保数据立即写入磁盘
                
        except Exception as e:
            print(f"处理batch {i//batch_size}时出错: {str(e)}")
            # 可以选择记录错误并继续处理下一个batch
            continue