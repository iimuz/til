/// @file

#include <iostream>

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
  int H;
  int W;
  int h;
  int w;
  is >> H >> W;
  is >> h >> w;

  const int restH = H - h;
  const int restW = W - w;
  const int restCell = restH * restW;

  os << restCell << "\n";

  return true;
}
}  // unnamed namespace
