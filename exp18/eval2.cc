#include "../json.hpp"
#include <iostream>
#include <fstream>
#include <regex>
#include <cctype>
#include <print>

using json = nlohmann::json;

int main(int argc, char** argv) {
  std::string input_filename1(argv[1]), input_filename2(argv[2]), answer_filename(argv[3]), group_filename1(std::format("./groups/{}", input_filename1)), group_filename2(std::format("./groups/{}", input_filename2));
  // std::println("filename:{}", input_filename1);
  std::map<int, int> answer;
  std::ifstream input_file1(input_filename1), answer_file(answer_filename), group_file1(group_filename1), input_file2(input_filename2), group_file2(group_filename2);
  std::map<int, std::string> group1, group2;
  for (std::string line; std::getline(group_file1, line); ) {
    auto data = json::parse(line);
    group1[data["pid"]] = data["result"];
  }
  for (std::string line; std::getline(group_file2, line); ) {
    auto data = json::parse(line);
    group2[data["pid"]] = data["result"];
  }
  for (std::string line; std::getline(answer_file, line); ) {
    auto data = json::parse(line);
    answer[data["pid"]] = data["answer"];
  }
  std::map<int, bool> is_correct1, is_correct2;
  for (std::string line; std::getline(input_file1, line); ) {
    auto data = json::parse(line);
    int pid = data["pid"];
    std::string content = data["content"];
    bool is_correct = false;
    // std::cerr << pid << '\n';
    if (std::smatch match; std::regex_search(content, match, std::regex(R"(\\boxed\{([^}]*)\})"))) {
      std::string number = match[1];
      try {
        if (std::stoll(number) == answer[pid]) {
          is_correct = true;
        }
      } catch (...) {}
    }
    is_correct1[pid] = is_correct;
  }
  for (std::string line; std::getline(input_file2, line); ) {
    auto data = json::parse(line);
    int pid = data["pid"];
    std::string content = data["content"];
    bool is_correct = false;
    // std::cerr << pid << '\n';
    if (std::smatch match; std::regex_search(content, match, std::regex(R"(\\boxed\{([^}]*)\})"))) {
      std::string number = match[1];
      try {
        if (std::stoll(number) == answer[pid]) {
          is_correct = true;
        }
      } catch (...) {}
    }
    is_correct2[pid] = is_correct;
  }
  int cnt = 0;
  for (auto [pid, cat] : group1) if (cat == "C" && group2[pid] == "A" && !is_correct1[pid] && is_correct2[pid]) {
    ++cnt;
  }
  std::println("{}", cnt);
  // std::ofstream out(output_filename, std::ios::app);
  // json output_json;
  // output_json["filename"] = input_filename;
  // output_json["correct"] = correct_cnt;
  // output_json["correct_rate"] = 1.0 * correct_cnt / tot;
  // out << output_json.dump() << '\n';
}