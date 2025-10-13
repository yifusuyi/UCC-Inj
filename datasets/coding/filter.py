import json

with open('problemset_en.jsonl', 'r', encoding='utf-8') as infile, \
     open('problemset_en2.jsonl', 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        try:
            data = json.loads(line.strip())
            tags = data.get('tags', [])
            
            # 检查 tags 是否同时包含两个目标字符串
            if '语言月赛' in tags and '2025' in tags:
                # 确保中文正常显示
                json_line = json.dumps(data, ensure_ascii=False)
                outfile.write(json_line + '\n')
            elif True:
                json_line = json.dumps(data, ensure_ascii=False)
                outfile.write(json_line + '\n')
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"处理行时出错: {e} | 内容: {line[:50]}...")

print("处理完成！符合条件的行已保存到 b.jsonl")