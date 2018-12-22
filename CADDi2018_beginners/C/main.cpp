/// @file

#include <cmath>
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
  typedef long Num_t;

  Num_t n;
  Num_t p;
  is >> n >> p;

  // n 乗したときに p を超えない最大数を求める
  Num_t maxNum = static_cast<Num_t>(std::ceil(std::pow(p, 1.0 / n)));

  // 最大数から逆側にあまりがなくなる値を求める
  Num_t divisor(maxNum);
  for (; divisor > 0; --divisor) {
    if (p % static_cast<Num_t>(std::pow(divisor, n)) == 0) break;
  }

  os << divisor << '\n';

  return true;
}
}  // unnamed namespace
