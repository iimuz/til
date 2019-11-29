/// @file

#include <iostream>
#include <map>
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

///
/// @brief 一回分の処理を行います。
/// @param[in] input 初期の枚数
/// @param[out] output 自分の変更後の枚数
/// @return 相手に渡す枚数
///
int oneStep_(const int input, int& output)
{
  int dst(input);
  if (dst % 2 == 1) --dst;
  output = dst / 2;
  return output;
}

/// @brief 実行処理
bool run(std::istream& is, std::ostream& os)
{
  int a;
  int b;
  int k;
  is >> a >> b >> k;

  int* target = &a;
  int* partner = &b;
  for (int i = 0; i < k; ++i) {
    *partner += oneStep_(*target, *target);
    std::swap(target, partner);
  }

  os << a << " " << b << "\n";

  return true;
}
}  // unnamed namespace
