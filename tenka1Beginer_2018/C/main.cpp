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

typedef int val_t;
typedef std::vector<val_t> arr_t;

/// @brief 実行処理
bool run(std::istream& is, std::ostream& os)
{
  // 入力データの読み込み
  int n;
  is >> n;

  arr_t arr(n);
  for(auto& v: arr) is >> v;

  if (n == 2) {
    os << std::abs(arr.front() - arr.back()) << "\n";
    return true;
  }

  // 昇順ソート
  std::sort(arr.begin(), arr.end());

  if (n == 3) {
    val_t dst(0);
    if ((arr[0] - arr[1]) == 0) {
      dst += std::abs(arr[2] - arr[0]);
      dst += std::abs(arr[2] - arr[1]);
    } else {
      dst += std::abs(arr[0] - arr[1]);
      dst += std::abs(arr[0] - arr[2]);
    }
    os << dst << "\n";
    return true;
  }

  if ((arr[1] - arr[0]) < (arr[n - 1] - arr[n - 2])) std::reverse(arr.begin(), arr.end());

  arr_t::const_iterator current = arr.begin();
  arr_t::const_iterator target = arr.end() - 1;

  val_t dst(0);
  dst += std::abs(*target - *current);
  dst += std::abs(*(target - 1) - *current);

  ++current;
  while (std::distance(current, target) > 0) {
    dst += std::abs(*target - *current);

    if (std::distance(current, target - 2) < 0) break;
    dst += std::abs(*(target - 2) - *current);

    ++current;
    --target;
    if (std::distance(current, target) < 2) break;
  }

  os << dst << "\n";

  return true;
}
}  // unnamed namespace
