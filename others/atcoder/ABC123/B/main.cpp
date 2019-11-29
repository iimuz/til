/// @file

#include <algorithm>
#include <iostream>
#include <vector>

namespace {

bool run(std::istream& is, std::ostream& os);
}  // unnamed namespace

/// @brief エントリポイント
#ifdef _TEST
static int run()
#else
int main()
#endif
{
  try {
    if (run(std::cin, std::cout) == false) {
      std::cerr << "main funciton error.\n";
      return EXIT_FAILURE;
    }
  } catch (const std::exception& e) {
    std::cerr << "catch exception\n";
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

namespace {

/// @brief 実行処理
bool run(std::istream& is, std::ostream& os)
{
  const int MAX_DISHES(5);
  std::vector<int> dishes(MAX_DISHES);
  for (auto& v: dishes) is >> v;

  std::vector<int> diffMinues;
  for (const auto& v: dishes) diffMinues.emplace_back(v % 10);
  for (auto& v: diffMinues) v = (10 - v) % 10;
  const auto MAX_DIFF = std::max_element(diffMinues.begin(), diffMinues.end());

  int count(0);
  bool last(false);
  for (int i = 0; i < MAX_DISHES; ++i) {
    if ((last == false) && diffMinues[i] == *MAX_DIFF) {
      count += dishes[i];
      last = true;
      continue;
    }

    count += dishes[i] + diffMinues[i];
  }

  os << count << "\n";

  return true;
}
}  // unnamed namespace
