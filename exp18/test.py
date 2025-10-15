from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import json
import os
import sys
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


input_model, input_filename = sys.argv[1], sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]
output_filename = f'groups/{input_model}/{input_filename}.jsonl'
Thinking=False
tp_cnt = len(sys.argv[3].split(','))


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
  tp=tp_cnt,            # 使用全部8张A800
  session_len=20000,      # 16K上下文长度
  max_batch_size=9,       # 保守批处理大小（确保精度）
  cache_max_entry_count=0.85,   # KV缓存比例
  quant_policy=0,         # 禁用量化（FP16精度）
  rope_scaling_factor=2.0,    # RoPE扩展因子
  use_logn_attn=True,       # 对数位置编码
  continuous_batching=True,   # 连续批处理
  enable_prefix_caching=True,   # 前缀缓存（优化长提示）
  max_prefill_token_num=200,   # 最大预填充token数
)
gen_config = GenerationConfig(do_sample=False,max_new_tokens=200)

def generate_llm_response(prompts):
  results = pipe(prompts, gen_config=gen_config)
  return results

finished_pid = set()
pid_list = []

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
with open("../datasets/gsm8k/problemset_encoded0.jsonl", "r", encoding="utf-8") as inf:
  lines = inf.readlines()
  for line in lines:
    data = json.loads(line)
    problems[data['pid']] = data['question']
model_output={}
with open(f"{input_model}/{input_filename}.jsonl", "r", encoding="utf-8") as inf:
  lines = inf.readlines()
  for line in lines:
    data = json.loads(line)
    model_output[data['pid']] = f'{data["thinking_content"]}\n{data["content"]}'

def gen_input(pid : int):
  if len(model_output[pid]) > 10000:
    return '''
    请你输出一行一个字符串"Too long input"，不要输出其他任何内容！/nothink
    '''
  user_content = rf''' 
    我正在进行一个和大语言模型能力评估相关的实验。具体来说，我会给模型一个混有噪声的问题，模型必须能够正确理解问题然后进行回答。
    模型的回答分为如下几种：
    [A] 模型在回答的过程中输出了正确的不含噪声的文本（意思对即可），然后给出了解题过程答案（无论答案是否正确）。
    [B] 模型在回答的过程中尝试输出了不含噪声的文本，但是给出的去噪文本的意思和原文本相比有错误（或缺失了关键信息），然后给出了解题过程和答案（无论答案是否正确）。
    [C] 模型完全没有显式地输出不含噪声的文本，而是直接给出了解题过程和答案（或没有解题过程只有答案）。
    注意，输出不含噪声的文本是指输出英文原文本。如果直接给出翻译后的文本，也视为情况 [C]。
    以下是对于输入问题：A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?的一个例子：
    [A] 类回答：我们需要先去除文本中的无意义字符，得到清晰的问题内容。\n\n原始文本是：\n\n```\n A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?\n```\n 问题翻译为：\n\n> 一根绳子需要 2 根蓝色纤维的螺栓，以及一半数量的白色纤维的螺栓。总共需要多少根螺栓？\n\n解答：\n\n- 蓝色纤维：2 根\n- 白色纤维：2 的一半，即 1 根\n- 总共：2 + 1 = **3**\n\n答案是：\n\n$$\n\\boxed{3}\n$$
    [B] 类回答：我们需要先清理文本中的无意义字符，得到清晰的句子。\n\n原始文本（去除无意义字符后）为：\n\n> A robot takes 2 balls of blue and a half of a red ball. How many balls in total does it take?\n\n翻译并理解问题：\n\n一个机器人拿走了 2 个蓝色球和 0.5 个红色球。问它一共拿了多少个球？\n\n计算过程：\n\n- 蓝色球：2 个  \n- 红色球：0.5 个  \n- 总数：2 + 0.5 = 2.5 个\n\n但题目要求答案是**整数**，因此我们判断是否题目中存在理解偏差。\n\n再仔细看原文：“a half of a red ball” —— 也就是 0.5 个红球，但题目问的是“how many balls”，即“多少个球”，而“0.5 个球”不是整数。\n\n因此，可能题目中“a half of a red ball”是干扰项，或者题目实际想问的是“how many whole balls”，即“多少个完整的球”。\n\n如果是这样，那么答案就是：2 个完整的球。\n\n最终答案是：$\\boxed{2}$
    [C] 类回答：一件长袍需要2匹蓝纤维和一半的白纤维，即$ \frac{2}{2} = 1$匹白纤维。\n\n总匹数 = 2（蓝）+ 1（白）= 3

    我将提供给你**没有噪声**的原文本和模型给出的回答，你需要帮我统计模型的输出是上面三类中的哪一类。你的输出只能有一个字符，是 A,B,C 之一，不能包含其他任何内容。
    原文本是：{problems[pid]}
    模型的回答是：{model_output[pid]}
    注意，你的输出必须只是 A,B,C 三个字符之一。
    /nothink
  '''
  return user_content

pipe = pipeline('/mnt/public/zay/Qwen3-30B-A3B',
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
    batch_prompts.append(gen_input(pid))
  
  # 生成响应
  try:
    results = pipe(batch_prompts, gen_config=gen_config)
    
    # 收集结果
    for j, result in enumerate(results):
  
      result_data = {
        'pid': batch_pids[j], 
        'result':split_think_content(result.text)["content"]
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