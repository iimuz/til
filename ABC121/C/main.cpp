/// @file

#include <algorithm>
#include <iomanip>
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
  long n;
  long m;
  is >> n >> m;

  std::vector<std::pair<long, long>> shopList(n);
  for (auto& v: shopList) is >> v.first >> v.second;

  std::sort(shopList.begin(), shopList.end());

  long sumMoney(0);
  long numDrinks(0);
  for (const auto& shop: shopList) {
    if (numDrinks + shop.second >= m) {
      const long BUY_NUM = m - numDrinks;
      sumMoney += shop.first * BUY_NUM;
      break;
    }

    numDrinks += shop.second;
    sumMoney += shop.first * shop.second;
  }

  os << sumMoney << "\n";

  return true;
}
}  // unnamed namespace
