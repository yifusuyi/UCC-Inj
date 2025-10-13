import random
import json


def insert_code(base: str, bytes_list: list[int]) -> str:
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
  encoded = []
  if text == None:
    return ''
  for char in text:
    #if (not char.isalpha()) and (not char.isdigit()):

    # 生成随机字节列表（长度随机，值随机）
    num_bytes = random.randint(min_bytes, max_bytes)
    random_bytes = [random.randint(0, 255) for _ in range(num_bytes)]

    # 生成含隐藏编码的字符并加入结果
    encoded_char = insert_code(char, random_bytes)
    encoded.append(encoded_char)

  return ''.join(encoded)


for inserted_cnt in range(0, 4):
  input_file_name = "problemset_en.jsonl"
  output_file_name = "problemset_encoded" + str(inserted_cnt) + ".jsonl"

  with open(input_file_name, "r", encoding="utf-8") as inf, \
       open(output_file_name, "w", encoding="utf-8") as outf:
    lines = inf.readlines()
    for line in lines:
      data = json.loads(line)
     # print(data)
      formed = data
      keyItems = ['background', 'description', 'inputFormat', 'outputFormat', 'hint']
      for key, value in data.items():
        if key in keyItems:
          formed[key] = insert_random_code(value, inserted_cnt, inserted_cnt)
        elif key == 'samples':
          samples = list()
          for sample in value:
            samples.append([insert_random_code(sample[0], inserted_cnt, inserted_cnt), 
                            insert_random_code(sample[1], inserted_cnt, inserted_cnt)])
          formed[key] = samples
        else:
          formed[key] = value
      outf.write(json.dumps(formed, ensure_ascii=False) + '\n')
      outf.flush()