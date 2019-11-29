/// @file

#include <cmath>
#include <iostream>
#include <map>
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
  int n;
  int m;
  int c;
  is >> n >> m >> c;

  std::vector<int> bList(m);
  for (auto& v: bList) is >> v;

  int correctAnswerNum(0);
  for (int i = 0; i < n; ++i) {
    std::vector<int> aList(m);
    for (auto& v: aList) is >> v;

    int sum(c);
    for (int j = 0; j < m; ++j) sum += aList[j] * bList[j];

    if (sum > 0) ++correctAnswerNum;
  }

  os << correctAnswerNum << "\n";

  return true;
}
}  // unnamed namespace
