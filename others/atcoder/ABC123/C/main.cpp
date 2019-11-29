/// @file

#include <algorithm>
#include <cmath>
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
  const int CAPACITY_NUM(5);
  unsigned long n;
  std::vector<unsigned long> capacity(CAPACITY_NUM);

  is >> n;
  for (auto& v: capacity) is >> v;

  const unsigned long MIN_CAPACITY = *std::min_element(capacity.begin(), capacity.end());
  const unsigned long MAX_TRANSPORT = std::ceil(static_cast<double>(n) / MIN_CAPACITY) - 1;
  const unsigned long MIN_TIME = MAX_TRANSPORT + CAPACITY_NUM;

  os << MIN_TIME << "\n";

  return true;
}
}  // unnamed namespace
