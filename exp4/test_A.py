from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import re
import sys
from tqdm import tqdm
from typing import List, Dict

model_path = sys.argv[1]  # 修改模型路径
# 从命令行参数获取输入输出文件路径和thinking标志
input_filename = sys.argv[2]
output_filename = sys.argv[3]
thinking = sys.argv[4].lower() == 'true'

# 1. 加载原始模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

# 2. 构建可打印 ASCII 字符到 token ID 的映射表
def build_char_to_token_map(tokenizer):
    """创建可打印 ASCII 字符到 token ID 的映射"""
    char_to_token = {}
    
    # 遍历所有可打印 ASCII 字符 (32-126)
    for ascii_val in range(32, 127):
        char = chr(ascii_val)
        
        # 特殊处理空格 - 在 Qwen 中通常表示为 "▁"
        if char == " ":
            token_id = tokenizer.convert_tokens_to_ids("▁")
            if token_id != tokenizer.unk_token_id:
                char_to_token[char] = token_id
                continue
        
        # 直接编码字符
        tokens = tokenizer.encode(char, add_special_tokens=False)
        
        # 如果字符被拆分为单个 token
        if len(tokens) == 1:
            char_to_token[char] = tokens[0]
        # 如果字符被拆分为多个 token (如某些标点符号)
        elif tokens:
            # 使用第一个 token 作为主要映射
            char_to_token[char] = tokens[0]
    
    # 添加换行符等特殊字符
    special_chars = {"\n": "▁", "\t": "▁", "\r": "▁"}
    for char, replacement in special_chars.items():
        token_id = tokenizer.convert_tokens_to_ids(replacement)
        if token_id != tokenizer.unk_token_id:
            char_to_token[char] = token_id
    
    return char_to_token

# 构建字符映射表
char_to_token = build_char_to_token_map(tokenizer)
print(f"已创建字符映射表，包含 {len(char_to_token)} 个字符的映射")

# 3. 创建应用聊天模板的函数（只对problem部分字符级分词）
def apply_char_level_chat_template(messages: List[Dict], char_map: Dict, thinking: bool = True):
    """应用聊天模板，只对problem部分进行字符级分词"""
    # 确保是单轮对话
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("只支持单轮用户对话")
    
    # 提取prompt和problem
    if not isinstance(messages[0]["content"], dict) or "prompt" not in messages[0]["content"] or "problem" not in messages[0]["content"]:
        raise ValueError("消息内容必须是包含'prompt'和'problem'键的字典")
    
    prompt_text = messages[0]["content"]["prompt"]
    problem_text = messages[0]["content"]["problem"]
    full_content = prompt_text + problem_text
    
    # 创建临时消息用于生成模板
    temp_messages = [{"role": "user", "content": full_content}]
    
    # 应用原始聊天模板（不进行分词）
    template_str = tokenizer.apply_chat_template(
        temp_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 使用正则表达式定位用户消息内容
    pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
    match = re.search(pattern, template_str, re.DOTALL)
    
    if not match:
        raise ValueError("无法在模板中找到用户消息内容")
    
    # 提取用户消息内容
    user_content = match.group(1)
    start_idx, end_idx = match.span(1)
    
    # 检查内容是否匹配
    if user_content != full_content:
        # 尝试去除可能的尾随空格
        if user_content.strip() == full_content.strip():
            user_content = user_content.strip()
            full_content = full_content.strip()
        else:
            raise ValueError(f"模板中的用户消息内容与输入不匹配\n模板内容: '{user_content}'\n输入内容: '{full_content}'")
    
    # 分割模板字符串
    prefix = template_str[:start_idx]
    suffix = template_str[end_idx:]
    
    # 如果不启用思考，在suffix中添加/nothink
    if not thinking:
        suffix = '/nothink' + suffix
    
    # 使用正常分词处理前缀
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    
    # 对prompt部分使用正常分词
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # 对problem部分进行字符级分词
    problem_char_ids = []
    for char in problem_text:
        if char in char_map:
            problem_char_ids.append(char_map[char])
        else:
            # 对于不在映射表中的字符，使用原始分词器处理
            tokens = tokenizer.encode(char, add_special_tokens=False)
            problem_char_ids.extend(tokens)
    
    # 使用正常分词处理后缀
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    
    # 组合所有 token IDs
    input_ids = prefix_ids + prompt_ids + problem_char_ids + suffix_ids
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    return input_ids

# 4. 分割思考内容和正式内容
def split_think_content(text: str) -> Dict:
    """将包含<think>标签的文本分割为思考内容和正式内容"""
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


# 5. 单批生成函数（修改为左填充）
def generate_batch(batch_messages: List[List[Dict]]) -> List[str]:
    """单批生成函数（左填充版本）"""
    # 为每个消息构建输入ID
    batch_input_ids = []
    input_lengths = []
    
    for messages in batch_messages:
        input_ids = apply_char_level_chat_template(messages, char_to_token, thinking)
        input_lengths.append(len(input_ids))
        batch_input_ids.append(input_ids)
    
    # 左填充批次
    max_len = max(len(ids) for ids in batch_input_ids)
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids in batch_input_ids:
        pad_len = max_len - len(ids)
        # 左填充：在序列开头添加填充token
        padded_ids = [tokenizer.pad_token_id] * pad_len + ids
        # 创建注意力掩码：填充部分为0，实际内容为1
        attention_mask = [0] * pad_len + [1] * len(ids)
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(attention_mask)
    
    # 转换为张量
    input_tensor = torch.tensor(padded_input_ids).to(model.device)
    attention_mask = torch.tensor(padded_attention_masks).to(model.device)
    
    # 生成输出（添加pad_token_id参数）
    outputs = model.generate(
        input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=1700,
        do_sample=False,
        temperature=0,
        pad_token_id=tokenizer.pad_token_id  # 明确指定填充token
    )
    
    # 处理每个样本的生成结果
    results = []
    for j in range(len(outputs)):
        # 提取生成部分（移除输入和左侧填充）
        # 注意：这里需要跳过左侧填充和原始输入
        start_idx = max_len  # 跳过整个输入部分（包括填充）
        output_ids = outputs[j][start_idx:]
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        results.append(generated_text)
    
    return results

# 6. 主处理函数
def process_dataset(input_file: str, output_file: str, thinking: bool = True, batch_size: int = 8):
    """处理整个数据集，每处理完一个batch就保存结果"""
    # 读取已完成的问题ID
    finished_pids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    finished_pids.add(data["pid"])
                except:
                    continue
    
    # 读取输入文件并过滤已完成的问题
    messages_list = []
    pid_list = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                pid = data["pid"]
                
                # 跳过已完成的问题
                if pid in finished_pids:
                    continue

                
                # 构建提示
                prompt_text = (
                    "你要解决这个问题，把答案放在 \\boxed{} 里：\n"
                    "问题的答案总是整数。\\boxed{} 的内容是评判你答案正确性的唯一依据，它必须仅包含一个整数。\n"
                    "例如，如果答案是 10000，你需要回答 \\boxed{10000}。回答 \\boxed{10,000}、\\boxed{1e4}、\\boxed{$10000$} 等都不得分。\n"
                    "题目如下：\n"
                )
                
                # 构建消息
                messages = [{
                    "role": "user",
                    "content": {
                        "prompt": prompt_text,
                        "problem": data["question"]
                    }
                }]
                
                messages_list.append(messages)
                pid_list.append(pid)
            except Exception as e:
                print(f"处理行时出错: {line.strip()}, 错误: {e}")
    
    if not messages_list:
        print("没有需要处理的新问题")
        return
    
    print(f"开始处理 {len(messages_list)} 个新问题...")
    
    # 分批处理并保存结果
    for i in tqdm(range(0, len(messages_list), batch_size), desc="Generating"):
        batch_start = i
        batch_end = min(i + batch_size, len(messages_list))
        
        batch_messages = messages_list[batch_start:batch_end]
        batch_pids = pid_list[batch_start:batch_end]
        
        try:
            # 生成当前batch的结果
            generated_texts = generate_batch(batch_messages)
            
            # 写入当前batch的结果
            with open(output_file, "a", encoding="utf-8") as f:
                for pid, text in zip(batch_pids, generated_texts):
                    result = split_think_content(text)
                    result["pid"] = pid
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()  # 确保立即写入磁盘
                    
        except Exception as e:
            print(f"处理批次 {i//batch_size} 时出错: {e}")
            # 可以选择记录失败的批次，稍后重试
            continue
        
    print(f"处理完成，结果已写入 {output_file}")

# 7. 执行主函数
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python script.py <input_file> <output_file> <thinking:true/false>")
        sys.exit(1)
    
    process_dataset(input_filename, output_filename, thinking)