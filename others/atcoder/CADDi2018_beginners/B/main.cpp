/// @file

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
  int n;
  int h;
  int w;
  is >> n >> h >> w;

  std::vector<std::pair<int, int>> ab(n);
  for (int i = 0; i < n; ++i) is >> ab[i].first >> ab[i].second;

  int count(0);
  for (const auto& v: ab) {
    if (v.first < h) continue;
    if (v.second < w) continue;
    ++count;
  }

  os << count << "\n";

  return true;
}
}  // unnamed namespace
