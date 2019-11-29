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

///
/// @brief 先頭から最後までが特定の文字列の繰り返しになっているか探索
/// @return 繰り返し構造を持つか否か
/// @retval true 繰り返しになっている
/// @retval false 繰り返しになっていない
///
bool searchNext_(
    const std::string::iterator begin,
    const std::string::iterator end)
{
  static const std::vector<std::string> SEARCH_STR {
    "dream",
    "dreamer",
    "erase",
    "eraser",
  };

  for (const auto& v: SEARCH_STR) {
    if (std::distance(begin, end) < v.size()) continue;

    const auto next = begin + v.size();
    if (v != std::string(begin, next)) continue;

    if (next == end) return true;
    if (searchNext_(next, end)) return true;
  }

  return false;
}

/// @brief 実行処理
bool run(std::istream& is, std::ostream& os)
{
  static const std::string YES("YES");
  static const std::string NO("NO");

  std::string s;
  is >> s;

  const bool SUCCESS = searchNext_(s.begin(), s.end());
  const std::string& RETURN_STR = SUCCESS ? YES : NO;

  os << RETURN_STR << "\n";

  return true;
}
}  // unnamed namespace
