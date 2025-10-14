#include "../json.hpp"
#include <iostream>
#include <fstream>
#include <regex>
#include <cctype>

using json = nlohmann::json;

int main(int argc, char** argv) {
  std::string input_filename(argv[1]), answer_filename(argv[2]), output_filename(argv[3]);
  std::map<int, int> answer;
  std::ifstream input_file(input_filename), answer_file(answer_filename);
  for (std::string line; std::getline(answer_file, line); ) {
    auto data = json::parse(line);
    answer[data["pid"]] = data["answer"];
  }
  int correct_cnt = 0, tot = 0;
  for (std::string line; std::getline(input_file, line); ) {
    auto data = json::parse(line);
    int pid = data["pid"];
    // std::cerr << pid << '\n';
    std::string content = data["content"];
    if (std::smatch match; std::regex_search(content, match, std::regex(R"(\\boxed\{([^}]*)\})"))) {
      std::string number = match[1];
      try {
        if (std::stoll(number) == answer[pid]) {
          ++correct_cnt;
        }
      } catch (...) {}
    }
    ++tot;
  }
  std::ofstream out(output_filename, std::ios::app);
  json output_json;
  output_json["filename"] = input_filename;
  output_json["correct"] = correct_cnt;
  output_json["correct_rate"] = 1.0 * correct_cnt / tot;
  out << output_json.dump() << '\n';
}