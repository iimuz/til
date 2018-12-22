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
      Node{"example1", "3 5 2\n10 3\n5 2\n2 5\n", "2\n"},
      Node{
      "example2",
      "10 587586158 185430194\n"
      "894597290 708587790\n"
      "680395892 306946994\n"
      "590262034 785368612\n"
      "922328576 106880540\n"
      "847058850 326169610\n"
      "936315062 193149191\n"
      "702035777 223363392\n"
      "11672949 146832978\n"
      "779291680 334178158\n"
      "615808191 701464268\n",
      "8\n"
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
