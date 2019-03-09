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
      Node{
        "example1",
        "2 3 -10\n1 2 3\n3 2 1\n1 2 2\n",
        "1\n"
      },
      Node{
        "example2",
        "5 2 -4\n-2 5\n100 41\n100 40\n-3 0\n-6 -2\n18 -13\n",
        "2\n"
      },
      Node{
        "example3",
        "3 3 0\n100 -100 0\n0 100 100\n100 100 100\n-100 100 100\n",
        "0\n"
      }
    )
);

namespace {

/// @brief テストケース名出力用
std::ostream& operator<<(std::ostream& os, const Node& n)
{
  return os << n.name;
}
}  // unnamed namespace
