from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import json
import os
import sys
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
input_filename, noise_cnt, shot_cnt, thinking = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].lower() == 'true'
output_filename = f'Llama3.1-70B/e{noise_cnt}s{shot_cnt}.jsonl'

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
    session_len=40535,            # 16K上下文长度
    max_batch_size=9,             # 保守批处理大小（确保精度）
    cache_max_entry_count=0.85,   # KV缓存比例
    quant_policy=0,               # 禁用量化（FP16精度）
    rope_scaling_factor=2.0,      # RoPE扩展因子
    use_logn_attn=True,           # 对数位置编码
    continuous_batching=True,     # 连续批处理
    enable_prefix_caching=True,   # 前缀缓存（优化长提示）
    max_prefill_token_num=20000,   # 最大预填充token数
)
gen_config = GenerationConfig(do_sample=False,max_new_tokens=20000)

def generate_llm_response(prompts):
    results = pipe(prompts, gen_config=gen_config)
    return results

finished_pid = set()
pid_list = []
my_prompts = []
shot_filename = 'shots.jsonl'
shot_prefix = []

# 读取已完成的pid
if os.path.exists(output_filename):
    with open(output_filename, "r", encoding='utf-8') as inf:
        lines = inf.readlines()
        for line in lines:
            try:
                data = json.loads(line)
                finished_pid.add(data['pid'])
            except:
                continue

problems = {}
with open(input_filename, "r", encoding="utf-8") as inf:
    lines = inf.readlines()
    for line in lines:
        data = json.loads(line)
        problems[data['pid']] = data['question']

def gen_input(pid : int):
    user_content = ''' 
         I will ask you a problem, and you must answer it. The answer to this problem is always an integer.  
      The text of this problem has been corrupted by noise, containing many invisible and meaningless characters. You need to remove these meaningless characters, then understand the text, and finally answer the question.  

      You must place your answer within \\boxed{}. The content of \\boxed{} is the sole basis for evaluating the correctness of your answer, and it must contain only a single integer.  
      For example, if the answer is 10000, you should respond with \\boxed{10000}. Answers such as \\boxed{10,000}, \\boxed{1e4}, or \\boxed{$10000$} will not be scored.
      The problem is：\n'''
    user_content += problems[pid]
    if not thinking:
        user_content += '/nothink'
    return {
        'role': 'user',
        'content': user_content
    }

with open(shot_filename, "r", encoding="utf-8") as inf:
    lines = inf.readlines()
    for line in lines:
        data = json.loads(line)
        pid = data['pid']
        if pid >= shot_cnt:
            break
        shot_prefix.append(gen_input(pid))
        shot_prefix.append({
            'role': 'assistant',
            'content': data['prefix'] + data["problem"] + data["suffix"]
        })

pipe = pipeline('meta-llama/Llama-3.1-70B-Instruct',
                backend_config=backend_config)

# 分批处理问题
batch_size = backend_config.max_batch_size
all_pids = [pid for pid in problems.keys() if pid not in finished_pid]

# 定义每多少批次保存一次结果
save_interval = 5

# 处理所有问题
results_to_save = []
for i in range(0, len(all_pids), batch_size):
    batch_pids = all_pids[i:i+batch_size]
    batch_prompts = []
    
    for pid in batch_pids:
        cur_dialog = []
        for item in shot_prefix:
            cur_dialog.append(item)
        cur_dialog.append(gen_input(pid))
        batch_prompts.append(cur_dialog)
    
    # 生成响应
    try:
        results = pipe(batch_prompts, gen_config=gen_config)
        
        # 收集结果
        for j, result in enumerate(results):
            response = split_think_content(result.text)
            result_data = {
                'pid': batch_pids[j], 
                'thinking_content': response['thinking_content'], 
                'content': response['content']
            }
            results_to_save.append(result_data)
        
        print(f"Processed batch {i//batch_size + 1}/{(len(all_pids)+batch_size-1)//batch_size}")
        
        # 每处理完save_interval个批次，保存一次结果
        if (i//batch_size + 1) % save_interval == 0 or i + batch_size >= len(all_pids):
            with open(output_filename, "a", encoding="utf-8") as outf:
                for result_data in results_to_save:
                    outf.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                outf.flush()  # 确保立即写入文件
                os.fsync(outf.fileno())  # 确保数据写入磁盘
            
            print(f"Saved {len(results_to_save)} results to file")
            results_to_save = []  # 清空结果列表
            
    except Exception as e:
        print(f"Error processing batch: {e}")
        # 保存已经处理的结果
        if results_to_save:
            with open(output_filename, "a", encoding="utf-8") as outf:
                for result_data in results_to_save:
                    outf.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                outf.flush()
                os.fsync(outf.fileno())
            results_to_save = []
        print("Saved partial results due to error")
        # 可以选择记录错误或跳过当前批次
        continue

print("Processing completed!")