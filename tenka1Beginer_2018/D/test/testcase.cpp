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

/// @brief テストケース
INSTANTIATE_TEST_CASE_P(
    normal,
    ExampleTest,
    testing::Values(
      // 今回の方法だと例題の順で検出しないため検出するパターンに修正
      // Node{"example1", "3\n", "Yes\n3\n2 1 2\n2 3 1\n2 2 3\n"},
      Node{"example1", "3\n", "Yes\n3\n2 1 3\n2 1 2\n2 2 3\n"},
      Node{"example2", "4\n", "No\n"},
      // AC しなかったのでテストケースを増やす
      Node{"example3", "5\n", "No\n"},
      Node{"example3", "6\n", "Yes\n4\n3 1 3 6\n3 1 2 4\n3 2 3 5\n3 4 5 6\n"},
      Node{"example3", "7\n", "No\n"},
      Node{"example3", "8\n", "No\n"},
      Node{"example3", "9\n", "No\n"},
      Node{"example3", "10\n", "Yes\n5\n4 1 3 6 10\n4 1 2 4 7\n4 2 3 5 8\n4 4 5 6 9\n4 7 8 9 10\n"}
    )
);

namespace {

/// @brief テストケース名出力用
std::ostream& operator<<(std::ostream& os, const Node& n)
{
  return os << n.name;
}
}  // unnamed namespace
