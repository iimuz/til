/// @file

#include <iostream>
#include <limits>
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
  using Numeric_t = unsigned long;
  const Numeric_t MAX_VALUE = 1e+12;
  Numeric_t a;
  Numeric_t b;
  is >> a >> b;

  unsigned long fxor(a);
  for (unsigned long i = a + 1; i <= b; ++i) {
    fxor ^= i;
  }

  os << fxor << "\n";

  return true;
}
}  // unnamed namespace
