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
      Node{"example1", "1\n2\n4\n8\n9\n15\n", "Yay!\n"},
      Node{"example2", "15\n18\n26\n35\n36\n18\n", ":(\n"}
    )
);

namespace {

/// @brief テストケース名出力用
std::ostream& operator<<(std::ostream& os, const Node& n)
{
  return os << n.name;
}
}  // unnamed namespace
