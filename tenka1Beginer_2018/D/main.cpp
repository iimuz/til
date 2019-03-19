/// @file

#include <cmath>
#include <iostream>
#include <map>
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
  using Numeric_t = int;

  Numeric_t n;
  is >> n;

  // 解の公式を利用して部分集合の k を求める
  // N は 1 以上のため、 k が正の整数になるためには、 + 側のみ
  const double K_PLUS = (1. + std::sqrt(1. + 8. * n)) / 2.;

  // k が整数でなければ部分集合を作れない
  if ((K_PLUS - std::floor(K_PLUS)) != 0.) {
    os << "No\n";
    return true;
  }

  // 部分集合を作成する
  const Numeric_t K(K_PLUS);
  std::map<Numeric_t, std::set<Numeric_t>> sub;
  for (Numeric_t i = 1, sum = 0; i < K; ++i) {
    sum += i;
    sub[0].insert(sum);
  }
  for (Numeric_t i = 1, sum = 1; i < K; sum += i, ++i) {
    Numeric_t count(sum);
    for (Numeric_t j = 0; j < K - 1; ++j, ++count) {
      sub[i].insert(count);
      if (sub[0].find(count) != sub[0].end()) break;
    }
    for (Numeric_t j = i, vsum = i; count <= n; ++j) {
      count += vsum;
      if (count > n) break;
      ++vsum;
      sub[i].insert(count);
    }
  }

  os << "Yes\n";
  os << K << "\n";
  for (const auto& s: sub) {
    os << s.second.size();
    for (const auto& v: s.second) os << " " << v;
    os << "\n";
  }

  return true;
}
}  // unnamed namespace
