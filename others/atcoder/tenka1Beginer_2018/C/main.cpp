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

using val_t = long;
using arr_t = std::vector<val_t>;

/// @brief 実行処理
bool run(std::istream& is, std::ostream& os)
{
  // 入力データの読み込み
  long n;
  is >> n;

  arr_t arr(n);
  for(auto& v: arr) is >> v;

  // 昇順ソート
  std::sort(arr.begin(), arr.end());

  val_t leftValue = arr.front();
  val_t rightValue = arr.back();
  val_t diffSum = rightValue - leftValue;
  for (std::size_t first = 1, last = arr.size() - 2; first <= last;) {
    const val_t ARR_FIRST = arr[first];
    const val_t ARR_LAST = arr[last];

    const val_t LEFT_FIRST = std::abs(leftValue - ARR_FIRST);
    const val_t LEFT_LAST = std::abs(leftValue - ARR_LAST);
    const val_t RIGHT_FIRST = std::abs(rightValue - ARR_FIRST);
    const val_t RIGHT_LAST = std::abs(rightValue - ARR_LAST);

    const val_t LEFT_MAX = std::max(LEFT_FIRST, LEFT_LAST);
    const val_t RIGHT_MAX = std::max(RIGHT_FIRST, RIGHT_LAST);
    if (LEFT_MAX > RIGHT_MAX) {
      diffSum += LEFT_MAX;
      if (LEFT_FIRST > LEFT_LAST) {
        leftValue = ARR_FIRST;
        ++first;
        continue;
      }

      leftValue = ARR_LAST;
      --last;
      continue;
    }

    diffSum += RIGHT_MAX;
    if (RIGHT_FIRST > RIGHT_LAST) {
      rightValue = ARR_FIRST;
      ++first;
      continue;
    }

    rightValue = ARR_LAST;
    --last;
  }

  os << diffSum << "\n";

  return true;
}
}  // unnamed namespace
