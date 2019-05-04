/// @file

#include "gtest/gtest.h"

namespace {

/// @brief テストの入出力パラメータ
struct Node {
  std::string name;
};  // struct Node

std::ostream& operator<<(std::ostream& os, const Node& n);

/// @brief 簡易テスト
class ExampleTest: public testing::TestWithParam<Node> {};
}  // unnamed namespace

/// @brief example test
TEST_P(ExampleTest, normal)
{
}

/// @brief テストケース
INSTANTIATE_TEST_CASE_P(
    normal,
    ExampleTest,
    testing::Values(
      Node{"example1"},
      Node{"example2"}
    )
);

namespace {

/// @brief テストケース名出力用
std::ostream& operator<<(std::ostream& os, const Node& n)
{
  return os << n.name;
}
}  // unnamed namespace
