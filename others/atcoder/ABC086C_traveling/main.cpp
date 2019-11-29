/// @file

#include <iostream>
#include <map>
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

/// @brief 時刻と場所を管理
struct TimePosition {
  int t;
  int x;
  int y;
};  // struct TimePosition

///
/// @brief 任意の位置までの最小距離を計算します
/// @param[in] sx 開始 x 座標
/// @param[in] sy 開始 y 座標
/// @param[in] ex 終了 x 座標
/// @param[in] ey 終了 y 座標
/// @return (sx, sy) から (ex, ey) までの距離
///
int calcMinDistance_(
    const int sx,
    const int sy,
    const int ex,
    const int ey)
{
  const int DIFF_X(std::abs(ex - sx));
  const int DIFF_Y(std::abs(ey - sy));
  return DIFF_X + DIFF_Y;
}

///
/// @brief 指定時間で動ける距離か判定
/// @param[in] time 時間
/// @param[in] dist 距離
/// @return 動けるかどうか
/// @retval true 移動可能
/// @retval false 移動できない
///
bool canArrive_(const int time, const int dist)
{
  bool arrive(true);
  if (time < dist) arrive = false;

  const int REST = time - dist;
  if (REST % 2 != 0) arrive = false;

  return arrive;
}

/// @brief 実行処理
bool run(std::istream& is, std::ostream& os)
{
  static const std::map<bool, std::string> OUTPUT_STR {
    {true, "Yes"},
    {false, "No"}
  };

  int n;
  is >> n;

  bool arrival(true);
  TimePosition current {0, 0, 0};
  for (int i = 0; i < n; ++i) {
    TimePosition next;
    is >> next.t >> next.x >> next.y;

    const int DISTANCE
      = calcMinDistance_(current.x, current.y, next.x, next.y);
    const int REST_TIME = next.t - current.t;
    arrival = canArrive_(REST_TIME, DISTANCE);
    if (arrival == false) break;

    current = std::move(next);
  }

  os << OUTPUT_STR.at(arrival) << "\n";

  return true;
}
}  // unnamed namespace
