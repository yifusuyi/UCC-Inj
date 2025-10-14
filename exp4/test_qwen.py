import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import re
import sys
from tqdm import tqdm
from pathlib import Path


def safe_open(filepath, mode='w', *args, **kwargs):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    return open(path, mode, *args, **kwargs)


def build_char_to_token_map(tokenizer):
    char_to_token = {}
    
    for ascii_val in range(32, 127):
        char = chr(ascii_val)
        
        if char == " ":
            token_id = tokenizer.convert_tokens_to_ids("▁")
            if token_id != tokenizer.unk_token_id:
                char_to_token[char] = token_id
                continue
        
        tokens = tokenizer.encode(char, add_special_tokens=False)
        
        if len(tokens) == 1:
            char_to_token[char] = tokens[0]
        elif tokens:
            char_to_token[char] = tokens[0]
    
    special_chars = {"\n": "▁", "\t": "▁", "\r": "▁"}
    for char, replacement in special_chars.items():
        token_id = tokenizer.convert_tokens_to_ids(replacement)
        if token_id != tokenizer.unk_token_id:
            char_to_token[char] = token_id
    
    return char_to_token

def apply_char_level_chat_template(messages, char_map, thinking=True):
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("只支持单轮用户对话")
    
    if not isinstance(messages[0]["content"], dict) or "prompt" not in messages[0]["content"] or "problem" not in messages[0]["content"]:
        raise ValueError("消息内容必须是包含'prompt'和'problem'键的字典")
    
    prompt_text = messages[0]["content"]["prompt"]
    problem_text = messages[0]["content"]["problem"]
    full_content = prompt_text + problem_text
    
    temp_messages = [{"role": "user", "content": full_content}]
    template_str = tokenizer.apply_chat_template(temp_messages, tokenize=False, add_generation_prompt=True)
    
    pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
    match = re.search(pattern, template_str, re.DOTALL)
    
    if not match:
        raise ValueError("无法在模板中找到用户消息内容")
    
    user_content = match.group(1)
    start_idx, end_idx = match.span(1)
    
    if user_content != full_content:
        if user_content.strip() == full_content.strip():
            user_content = user_content.strip()
            full_content = full_content.strip()
        else:
            raise ValueError(f"模板中的用户消息内容与输入不匹配\n模板内容: '{user_content}'\n输入内容: '{full_content}'")
    
    prefix = template_str[:start_idx]
    suffix = template_str[end_idx:]
    
    if not thinking:
        suffix = '/nothink' + suffix
    
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    problem_char_ids = []
    for char in problem_text:
        if char in char_map:
            problem_char_ids.append(char_map[char])
        else:
            tokens = tokenizer.encode(char, add_special_tokens=False)
            problem_char_ids.extend(tokens)
    
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    input_ids = prefix_ids + prompt_ids + problem_char_ids + suffix_ids
    
    return input_ids

def split_think_content(text):
    pattern = r'<think>(.*?)</think>\s*(.*)'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return {"thinking_content": match.group(1).strip(), "content": match.group(2).strip()}
    else:
        return {"thinking_content": None, "content": text.strip()}

def generate_batch(batch_messages, thinking):
    batch_input_ids = []
    input_lengths = []
    
    for messages in batch_messages:
        input_ids = apply_char_level_chat_template(messages, char_to_token, thinking)
        input_lengths.append(len(input_ids))
        batch_input_ids.append(input_ids)
    
    max_len = max(len(ids) for ids in batch_input_ids)
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids in batch_input_ids:
        pad_len = max_len - len(ids)
        padded_ids = [tokenizer.pad_token_id] * pad_len + ids
        attention_mask = [0] * pad_len + [1] * len(ids)
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(attention_mask)
    
    input_tensor = torch.tensor(padded_input_ids).to(model.device)
    attention_mask = torch.tensor(padded_attention_masks).to(model.device)
    
    outputs = model.generate(
        input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=1700,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    
    results = []
    for j in range(len(outputs)):
        start_idx = max_len
        output_ids = outputs[j][start_idx:]
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        results.append(generated_text)
    
    return results

def process_dataset(input_file, output_file, thinking=True, batch_size=8):
    finished_pids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    finished_pids.add(data["pid"])
                except:
                    continue
    
    messages_list = []
    pid_list = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                pid = data["pid"]
                
                if pid in finished_pids:
                    continue

                prompt_text = (
                    "你要解决这个问题，把答案放在 \\boxed{} 里：\n"
                    "问题的答案总是整数。\\boxed{} 的内容是评判你答案正确性的唯一依据，它必须仅包含一个整数。\n"
                    "例如，如果答案是 10000，你需要回答 \\boxed{10000}。回答 \\boxed{10,000}、\\boxed{1e4}、\\boxed{$10000$} 等都不得分。\n"
                    "题目如下：\n"
                )
                
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
    
    for i in tqdm(range(0, len(messages_list), batch_size), desc="Generating"):
        batch_start = i
        batch_end = min(i + batch_size, len(messages_list))
        
        batch_messages = messages_list[batch_start:batch_end]
        batch_pids = pid_list[batch_start:batch_end]
        
        try:
            generated_texts = generate_batch(batch_messages, thinking)
            
            with safe_open(output_file, mode = "a", encoding="utf-8") as f:
                for pid, text in zip(batch_pids, generated_texts):
                    result = split_think_content(text)
                    result["pid"] = pid
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    
        except Exception as e:
            print(f"处理批次 {i//batch_size} 时出错: {e}")
            continue
        
    print(f"处理完成，结果已写入 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理文本生成任务")
    parser.add_argument("input_file", help="输入文件路径")
    parser.add_argument("output_file", help="输出文件路径")
    parser.add_argument("thinking", type=lambda x: x.lower() == 'true', help="是否启用思考模式 (true/false)")
    parser.add_argument("--model_family", help="")
    parser.add_argument("--model_name", help="")
    parser.add_argument("--gpu_device", default="0,1,2,3,4,5,6,7", help="GPU设备ID")
    
    args = parser.parse_args()
  #  print(args.thinking, args.input_file, args.output_file)
   # exit(0)

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    model_name = f'{args.model_family}/{args.model_name}'
    output_file = f'{args.model_name}/{args.output_file}'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    char_to_token = build_char_to_token_map(tokenizer)
    print(f"已创建字符映射表，包含 {len(char_to_token)} 个字符的映射")
    process_dataset(args.input_file, output_file, args.thinking)