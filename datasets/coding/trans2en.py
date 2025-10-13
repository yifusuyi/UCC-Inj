# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import json

with open("ds_key.txt", "r", encoding="utf-8") as f:
  ds_key = f.read().strip()

client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")

input_file_name = "problemset.jsonl"
output_file_name = "problemset_en.jsonl"

with open(input_file_name, "r", encoding="utf-8") as inf, \
     open(output_file_name, "a", encoding="utf-8") as outf:
  lines = inf.readlines()
  for line in lines:
    data = json.loads(line)
    content = json.dumps(data)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": 
            "我将给你一个 json，这个 json 包含多个字段。其中 'background', 'description', 'inputFormat', 'outputFormat' 和 'Hint' 字段是中文。你需要把这五个字段的内容都翻译成英文。翻译应该保证尽量按原文的意思翻译，不能随意修改原文的意思，不能漏掉任何信息，也不能自行添加信息和总结。其他字段保持中文。你需要输出一行一个 json，这个 json 的各个字段名和输入 json 一致，被翻译的五个字段的内容是翻译后的内容，其他字段保持不变。你不能输出任何其他内容，输出必须是且仅是一个完整合法的 json，不要有换行。直接输出 json 本体而不要把 json 用 ```json 的标记括起来。"},
            {"role": "user", "content": content},
        ],
        stream=False
    )
    result = response.choices[0].message.content
    outf.write(result + "\n")
    outf.flush()