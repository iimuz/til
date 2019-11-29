/// @file

#include <algorithm>
#include <iostream>
#include <string>
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
  const std::string CONNECT_STR("Yay!");
  const std::string DISCONNECT_STR(":(");
  const int ANTENNA_NUM(5);
  std::vector<int> distances(ANTENNA_NUM);
  int k;
  for (auto& v: distances) is >> v;
  is >> k;

  std::sort(distances.begin(), distances.end());
  const int MAX_DISTANCE(distances.back() - distances.front());

  const std::string* anser(&CONNECT_STR);
  if (MAX_DISTANCE > k) anser = &DISCONNECT_STR;

  os << *anser << "\n";

  return true;
}
}  // unnamed namespace
