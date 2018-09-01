/// @file

#include <iostream>
#include <string>

namespace {

bool run(std::istream& is, std::ostream& os);
}  // unnamed namespace

/// @brief エントリポイント
int main()
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
  int a;
  is >> a;

  int b;
  int c;
  is >> b >> c;

  std::string s;
  is >> s;

  int sum(a + b + c);
  os << sum << " " << s << "\n";

  return true;
}
}  // unnamed namespace
