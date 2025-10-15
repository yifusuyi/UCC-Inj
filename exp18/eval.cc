#include "../json.hpp"
#include <iostream>
#include <fstream>
#include <regex>
#include <cctype>
#include <print>

using json = nlohmann::json;

int main(int argc, char** argv) {
  std::string input_filename(argv[1]), answer_filename(argv[2]), group_filename(std::format("./groups/{}", input_filename));
  std::println("filename:{}", input_filename);
  std::map<int, int> answer;
  std::ifstream input_file(input_filename), answer_file(answer_filename), group_file(group_filename);
  std::map<int, std::string> groups;
  for (std::string line; std::getline(group_file, line); ) {
    auto data = json::parse(line);
    groups[data["pid"]] = data["result"];
  }
  for (std::string line; std::getline(answer_file, line); ) {
    auto data = json::parse(line);
    answer[data["pid"]] = data["answer"];
  }
  int correct_cnt = 0, tot = 0, tot2 = 0;
  std::map<std::pair<std::string, std::string>, int> rec;
  for (std::string line; std::getline(input_file, line); ) {
    auto data = json::parse(line);
    int pid = data["pid"];
    std::string content = data["content"];
    bool is_correct = false;
    ++tot2;
    // std::cerr << pid << '\n';
    if (std::smatch match; std::regex_search(content, match, std::regex(R"(\\boxed\{([^}]*)\})"))) {
      std::string number = match[1];
      try {
        if (std::stoll(number) == answer[pid]) {
          is_correct = true;
        }
      } catch (...) {}
    }
    if (is_correct) ++rec[{groups[pid], "T"}];
    else ++rec[{groups[pid], "F"}];
    if (groups[pid].size() == 1) ++tot;
  }
  int acc_cnt = 0;
  
  for (auto [i, j] : rec) {
    if (i.second == "T") acc_cnt += j;;
    std::println("{} {} {:.2f}%", i.first, i.second, 100.0 * j / tot);
  }
  std::println("ACC {} {} {:.4f}%", acc_cnt, tot2, 1.0 * acc_cnt / tot2);
  std::println("\n\n");
  // std::ofstream out(output_filename, std::ios::app);
  // json output_json;
  // output_json["filename"] = input_filename;
  // output_json["correct"] = correct_cnt;
  // output_json["correct_rate"] = 1.0 * correct_cnt / tot;
  // out << output_json.dump() << '\n';
}