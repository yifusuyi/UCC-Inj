import random
import json


def insertcode(base: str, bytes_list: list[int]) -> str:
  """
  将字节列表编码为 Unicode 变体选择符并附加到基础字符后
  
  :param base: 单个小写字母
  :param bytes_list: 要隐藏的字节列表 (0-255)
  :return: 包含隐藏信息的字符串
  """
  encoded = [base]

  for byte in bytes_list:
    if byte < 16:
      codepoint = 0xFE00 + byte
    else:
      codepoint = 0xE0100 + (byte - 16)

    encoded.append(chr(codepoint))

  return ''.join(encoded)


def insert_random_code(text: str,
                       min_bytes: int = 0,
                       max_bytes: int = 1) -> str:
  """
    为文本每个字母后附加随机长度的不可见编码
    
    :param text: 纯英文文本（仅处理小写字母）
    :param min_bytes: 每个字母后最少附加字节数（默认1）
    :param max_bytes: 每个字母后最多附加字节数（默认3）
    :return: 含随机隐藏编码的字符串
    """
  encoded = []
  for char in text:
    '''if not char.islower():  # 仅处理小写字母
      encoded.append(char)
      continue'''

    # 生成随机字节列表（长度随机，值随机）
    num_bytes = random.randint(min_bytes, max_bytes)
    random_bytes = [random.randint(0, 255) for _ in range(num_bytes)]

    # 生成含隐藏编码的字符并加入结果
    encoded_char = insertcode(char, random_bytes)
    encoded.append(encoded_char)

  return ''.join(encoded)


for i in range(0, 4):
  input_file_name = "GSM8K_zh.jsonl"
  output_file_name = "problemset_encoded" + str(i) + ".jsonl"
  pid = 0
  with open(input_file_name, "r", encoding="utf-8") as inf, \
      open(output_file_name, "w", encoding="utf-8") as outf:
    lines = inf.readlines()
    for line in lines:
      data = json.loads(line)
      if data['split'] == 'train':
        continue
      processed_data = {}
      processed_data['pid'] = pid
      pid += 1
      contents = 'question_zh'
      original_text = data[contents]
      protected_text = insert_random_code(original_text, i, i)
      processed_data['question'] = protected_text
      answer = data['answer_only']
      processed_data['answer'] = int(answer.replace(',', ''))
      outf.write(json.dumps(processed_data, ensure_ascii=False) + "\n")
      outf.flush()
