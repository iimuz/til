/// @file

#include <iostream>
#include <set>
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
  // カードの読み取りとソート
  int n;
  is >> n;

  std::multiset<int, std::greater<int>> cardSet;
  for (int i = 0; i < n; ++i) {
    int a;
    is >> a;
    cardSet.insert(a);
  }

  // カードを昇順に交互に取得する
  int sumAlice(0);
  int sumBob(0);
  bool isAlice(true);
  for (const auto& v: cardSet) {
    if (isAlice) sumAlice += v;
    else sumBob += v;
    isAlice = !isAlice;
  }

  // 得点の差分を出力
  const int DIFF = sumAlice - sumBob;
  os << DIFF << "\n";

  return true;
}
}  // unnamed namespace
