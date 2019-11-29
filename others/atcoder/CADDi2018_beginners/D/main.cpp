/// @file

#include <algorithm>
#include <iostream>
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
  const std::string WINNER_YOU("first");
  const std::string WINNER_RUNRUN("second");

  long n;
  is >> n;

  std::vector<long> a(n);
  for (long i = 0; i < n; ++i) is >> a[i];

  auto itOdd = std::find_if(a.begin(), a.end(), [](long val) { return val % 2; });

  const std::string* WINNER = &WINNER_YOU;
  if (itOdd == a.end()) WINNER = &WINNER_RUNRUN;

  os << *WINNER << "\n";

  return true;
}
}  // unnamed namespace
