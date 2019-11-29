/// @file

#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
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
  int n;
  int m;
  is >> n >> m;

  // 県番号、 誕生年
  std::vector<std::pair<int, int>> cityList(m);
  for (auto& v: cityList) is >> v.first >> v.second;

  // 県番号、 誕生年、 市番号
  std::map<int, std::map<int, int>> prefList;
  for (std::size_t idx = 0; idx < cityList.size(); ++idx) {
    const auto& city = cityList[idx];
    prefList[city.first][city.second] = idx;
  }

  // map によって並び替えられた順に市番号を振っていく
  std::vector<std::string> cityNumbers(m);
  for (const auto& pref: prefList) {
    int num(1);
    for (const auto& city: pref.second) {
      std::ostringstream ss;
      ss << std::setw(6) << std::setfill('0') << pref.first;
      const std::string PREF_STR(ss.str());

      ss.str("");
      ss.clear(std::stringstream::goodbit);

      ss << std::setw(6) << std::setfill('0') << num;
      const std::string CITY_STR(ss.str());

      cityNumbers[city.second] = PREF_STR + CITY_STR;

      ++num;
    }
  }

  // 出力
  for (const auto& v: cityNumbers) os << v << "\n";

  return true;
}
}  // unnamed namespace
