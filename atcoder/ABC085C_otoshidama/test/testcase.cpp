/// @file

#define _TEST
#include "../main.cpp"
#undef _TEST

#include "gtest/gtest.h"

#include <sstream>
#include <string>

namespace {

/// @brief テストの入出力パラメータ
struct Node {
  std::string name;

  std::string input;
  std::string expect;
};  // struct Node

std::ostream& operator<<(std::ostream& os, const Node& n);

/// @brief 簡易テスト
class ExampleTest: public testing::TestWithParam<Node> {};
}  // unnamed namespace

/// @brief example test
TEST_P(ExampleTest, normal)
{
  std::stringstream sin(GetParam().input);
  std::stringstream sout("");
  run(sin, sout);

  ASSERT_EQ(GetParam().expect, sout.str());
}

///
/// @brief テストケース
/// @note 例題と違うタイプの回答を返すアルゴリズムのため回答例を変更
///
INSTANTIATE_TEST_CASE_P(
    normal,
    ExampleTest,
    testing::Values(
      // Node{"example1", "9 45000\n", "4 0 5\n"},
      Node{"example1", "9 45000\n", "0 9 0\n"},
      Node{"example2", "20 196000\n", "-1 -1 -1\n"},
      // Node{"example3", "1000 1234000\n", "14 27 959\n"}
      Node{"example3", "1000 1234000\n", "2 54 944\n"}
    )
);

namespace {

/// @brief テストケース名出力用
std::ostream& operator<<(std::ostream& os, const Node& n)
{
  return os << n.name;
}
}  // unnamed namespace
