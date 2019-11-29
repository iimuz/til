/// @file

#include <algorithm>
#include <iostream>
#include <string>

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
  static const char NONE('0');

  std::string buff;
  is >> buff;

  const int SUM = std::count_if(
      buff.begin(),
      buff.end(),
      [](const char& c) {
        if (c == NONE) return false;
        return true;
      }
  );

  os << SUM << "\n";

  return true;
}
}  // unnamed namespace
