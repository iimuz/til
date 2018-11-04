/// @file

#include <cmath>
#include <iostream>
#include <map>
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
  const double GRAD_TEMP(6E-3);

  int n;
  is >> n;

  int t;
  int a;
  is >> t >> a;

  std::vector<int> hArray(n);
  for (auto& v: hArray) is >> v;

  // 指定した気温との誤差を算出
  std::map<double, int> epsMap;
  for (std::size_t idx = 0; idx < hArray.size(); ++idx) {
    const double TEMP = static_cast<double>(t) - hArray[idx] * GRAD_TEMP;
    const double EPS = std::abs(static_cast<double>(a) - TEMP);
    epsMap[EPS] = idx;
  }

  os << epsMap.begin()->second + 1 << "\n";

  return true;
}
}  // unnamed namespace
