/// @file

#include <cmath>
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
  static const int COIN_GCD(50);
  static const int COIN_A(500 / COIN_GCD);
  static const int COIN_B(100 / COIN_GCD);
  static const int COIN_C(50 / COIN_GCD);

  int a;
  int b;
  int c;
  int x;
  is >> a >> b >> c >> x;

  const int VAL = x / COIN_GCD;
  int count(0);
  for (int ai = 0; ai <= a; ++ai) {
    const int REST_C = VAL - COIN_A * ai;
    if (REST_C < 0) break;

    for (int bi = 0; bi <= b; ++bi) {
      const int REST_B = REST_C - COIN_B * bi;
      if (REST_B < 0) break;
      if (c < REST_B) continue;

      ++count;
    }
  }


  os << count << "\n";

  return true;
}
}  // unnamed namespace
