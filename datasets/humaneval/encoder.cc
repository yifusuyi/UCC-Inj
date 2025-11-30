#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cctype>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include "../json.hpp"

using json = nlohmann::json;

// 将 Unicode 码点转换为 UTF-8 字符串
std::string codepoint_to_utf8(uint32_t codepoint) {
  if (codepoint <= 0x7F) {
    return {static_cast<char>(codepoint)};
  } else if (codepoint <= 0x7FF) {
    return {
      static_cast<char>(0xC0 | (codepoint >> 6)),
      static_cast<char>(0x80 | (codepoint & 0x3F))
    };
  } else if (codepoint <= 0xFFFF) {
    return {
      static_cast<char>(0xE0 | (codepoint >> 12)),
      static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)),
      static_cast<char>(0x80 | (codepoint & 0x3F))
    };
  } else {
    return {
      static_cast<char>(0xF0 | (codepoint >> 18)),
      static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)),
      static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)),
      static_cast<char>(0x80 | (codepoint & 0x3F))
    };
  }
}

// 在基础字符后插入隐藏编码
std::string insert_code(const std::string& base_char, const std::vector<uint8_t>& bytes_list) {
  std::string result = base_char;
  for (uint8_t byte : bytes_list) {
    uint32_t codepoint;
    if (byte < 16) {
      codepoint = 0xFE00 + byte;
    } else {
      codepoint = 0xE0100 + (byte - 16);
    }
    result += codepoint_to_utf8(codepoint);
  }
  return result;
}

// 将 UTF-8 字符串拆分为字符序列
std::vector<std::string> split_utf8(const std::string& str) {
  std::vector<std::string> characters;
  for (size_t i = 0; i < str.size(); ) {
    int len = 1;
    uint8_t c = static_cast<uint8_t>(str[i]);
    if (c >= 0xF0) len = 4;    // 4字节字符
    else if (c >= 0xE0) len = 3; // 3字节字符
    else if (c >= 0xC0) len = 2; // 2字节字符
    characters.push_back(str.substr(i, len));
    i += len;
  }
  return characters;
}

// 在文本中随机插入隐藏编码
std::string insert_random_code(const std::string& text, int min_bytes, int max_bytes, 
                               std::mt19937& rng, int byte_range=256) {
  if (text.empty()) return "";
  
  std::vector<std::string> chars = split_utf8(text);
  std::string result;
  std::uniform_int_distribution<int> num_bytes_dist(min_bytes, max_bytes);
  std::uniform_int_distribution<uint8_t> byte_dist(0, byte_range - 1);
  
  for (const auto& ch : chars) {
    int num_bytes = num_bytes_dist(rng);
    std::vector<uint8_t> random_bytes;
    for (int i = 0; i < num_bytes; ++i) {
      random_bytes.push_back(byte_dist(rng));
    }
    result += insert_code(ch, random_bytes);
  }
  return result;
}

int main() {
  std::random_device rd;
  std::mt19937 rng(rd());
  
  for (int i : {0,1,2,3,4}) {
    std::string input_file = "test.jsonl";
    std::string output_file = "problemset_encoded" + std::to_string(i) + ".jsonl";
    
    std::ifstream inf(input_file);
    std::ofstream outf(output_file);
    
    if (!inf.is_open() || !outf.is_open()) {
      continue; // 跳过无法打开的文件
    }
    
    std::string line;
    while (std::getline(inf, line)) {
      try {
        json data = json::parse(line);
        json processed_data = data;
        
        if (data.contains("prompt") && data["prompt"].is_string()) {
          auto original_prompt = data["prompt"].get<std::string>();
          processed_data["prompt"] = insert_random_code(original_prompt, i, i, rng);
        }

        outf << processed_data.dump() << '\n';
        outf.flush();
        
      } catch (const json::parse_error& err) {
        // 跳过解析错误行
      }
    }
  }
  
  return 0;
}