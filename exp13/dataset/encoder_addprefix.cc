#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cctype>
#include <algorithm>
#include <cstdint>
#include <format>
#include <iostream>
#include <ranges>
#include "../../json.hpp"

using json = nlohmann::json;

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
                               std::mt19937& rng) {
  if (text.empty()) return "";

  std::vector<std::string> chars = split_utf8(text);
  std::string result;
  std::uniform_int_distribution<int> num_bytes_dist(min_bytes, max_bytes);
  std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
  
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
  std::vector<std::string> commands;

  std::vector<std::pair<int, int>> insert_num_range {
    {0, 0}, {1, 1}, {2, 2}, {3, 3}
  };

  std::ifstream meaningless_prefix_file("meaningless_text.txt");
  if (!meaningless_prefix_file.is_open()) {
    std::cerr << "Failed to open meaningless_prefix.txt" << std::endl;
  }
  std::vector<std::string> meaningless_prefixes;
  for (std::string line; std::getline(meaningless_prefix_file, line); ) {
    if (!line.empty()) {
      meaningless_prefixes.push_back(line);
    }
  }
  assert(meaningless_prefixes.size() == 3);
  
  for (auto [lower, upper] : insert_num_range) {
    int prefix_id = 0;
    for (auto &prefix : meaningless_prefixes) {
      std::string input_file = "test.jsonl";
      if (lower != upper) {
        std::cerr << "lower not qual to upper!";
        return -1;
      }
      std::string output_file = std::format("problemset_encoded_{}_{}.jsonl", upper, prefix_id);
      std::string result_filename = std::format("e{}{}.jsonl", upper, prefix_id);
      prefix_id++;
      commands.push_back(std::format("python test_qwen3_30B.py ./dataset/{} qwen3_30B_normal/{} False", output_file, result_filename));
      commands.push_back(std::format("python test_qwen3_30B.py ./dataset/{} qwen3_30B_normal/{} True", output_file, result_filename));
      
      std::ifstream inf(input_file);
      std::ofstream outf(output_file);
      
      if (!inf.is_open() || !outf.is_open()) {
        std::cerr << "ovo, failed to open files: " << input_file << " or " << output_file << std::endl;
        continue;
      }
      
      int pid = 0;
      for (std::string line; std::getline(inf, line); ) {
        try {
          json data = json::parse(line);
          json processed_data;
          
          // 设置处理后的数据
          processed_data["pid"] = pid++;
          
          // 处理 question 字段
          if (data.contains("question") && data["question"].is_string()) {
            std::string question = prefix + data["question"].get<std::string>();
            processed_data["question"] = insert_random_code(question, lower, upper, rng);
          }
          
          // 处理 answer 字段
          if (data.contains("answer") && data["answer"].is_string()) {
            std::string answer_str = data["answer"].get<std::string>();
            
            // 查找最后一个 "#### " 分隔符
            size_t pos = answer_str.rfind("#### ");
            if (pos != std::string::npos) {
              // 提取分隔符后的数字部分
              answer_str = answer_str.substr(pos + 5);
            }
            
            // 移除数字中的逗号
            answer_str.erase(std::remove(answer_str.begin(), answer_str.end(), ','), answer_str.end());
            
            try {
              // 转换为整数
              long long answer_val = std::stoll(answer_str);
              processed_data["answer"] = answer_val;
            } catch (const std::exception&) {
              // 转换失败时设为0
              processed_data["answer"] = 0;
            }
          }
          // 写入处理后的数据
          outf << processed_data.dump() << '\n';
          outf.flush();
        } catch (const json::parse_error&) {
          // 跳过解析错误行
        }
      }
    }
  }
  std::ranges::sort(commands, [](auto a, auto &b) -> bool {
    char x1 = a[a.size() - 2], x2 = b[b.size() - 2];
    if (x1 != x2) return x1 == 's';
    return a < b;
  });
  std::ofstream cmdf("run_commands.sh");
  for (auto command : commands) {
    std::string output_content = command;
    if (command != commands.back()) output_content += " &&";
    std::println(cmdf, "{}", output_content);
  }
}