/// @file

#include <iostream>
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
  static const int GCD(1000);
  static const int NOGUCHI(1000 / GCD);
  static const int HIGUCHI(5000 / GCD);
  static const int YUKICHI(10000 / GCD);
  static const int INVALID_NUM(-1);

  int n;
  int y;
  is >> n >> y;

  const int TOTAL =  y / GCD;

  int numYukichi(INVALID_NUM);
  int numHiguchi(INVALID_NUM);
  int numNoguchi(INVALID_NUM);
  for (int numY = 0; numY <= n; ++numY) {
    if (numYukichi != INVALID_NUM) break;

    const int COST_YUKICHI = numY * YUKICHI;
    if (COST_YUKICHI > TOTAL) break;
    if ((COST_YUKICHI == TOTAL) && (numY == n)) {
      numYukichi = numY;
      numHiguchi = 0;
      numNoguchi = 0;
      break;
    }

    const int REST_NUM = n - numY;
    for (int numH = 0; numH <= REST_NUM; ++numH) {
      const int COST_HIGUCHI = numH * HIGUCHI;
      if ((COST_YUKICHI + COST_HIGUCHI) > TOTAL) break;

      const int NUM_NOGUCHI = TOTAL - (COST_YUKICHI + COST_HIGUCHI);
      const int TOTAL_NUM = numY + numH + NUM_NOGUCHI;
      if (TOTAL_NUM < n) break;
      if (TOTAL_NUM == n) {
        numYukichi = numY;
        numHiguchi = numH;
        numNoguchi = NUM_NOGUCHI;
        break;
      }
    }
  }

  os << numYukichi << " " << numHiguchi << " " << numNoguchi << "\n";

  return true;
}
}  // unnamed namespace
