/// @file

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

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
  using Vec_t = std::vector<Numeric_t>

  auto readF = [&is]() {
    Numeric_t v;
    is >> v;
    return v;
  };
  const Numeric_t X = readF();
  const Numeric_t Y = readF();
  const Numeric_t Z = readF();
  const Numeric_t K = readF();

  auto readVecF = [&is](const Numeric_t& n) {
    std::vector<Numeric_t> vec(n);
    for (auto& v: vec) is >> v;
    return std::move(vec);
  };
  Vec_t a = readVecF(X);
  Vec_t b = readVecF(Y);
  Vec_t c = readVecF(Z);

  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  std::sort(c.begin(), c.end());

  auto ita = a.crbegin();
  auto itb = b.crbegin();
  auto itc = c.crbegin();
  Vec_t delicious(K);
  for (auto v: delicious) {
  }

  os << 0 << "\n";

  return true;
}
}  // unnamed namespace
