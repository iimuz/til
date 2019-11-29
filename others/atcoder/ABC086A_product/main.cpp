/// @file

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
  static const std::string STR_ODD("Odd");
  static const std::string STR_EVEN("Even");

  int a;
  int b;
  is >> a >> b;

  bool isOdd(true);
  if ((a % 2) == 0) isOdd = false;
  if ((b % 2) == 0) isOdd = false;

  const std::string& STR_OUT = isOdd ? STR_ODD : STR_EVEN;
  os << STR_OUT << "\n";

  return true;
}
}  // unnamed namespace
