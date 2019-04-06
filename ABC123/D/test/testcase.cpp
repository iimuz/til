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
      Node{"example1", "2 2 2 8\n4 6\n1 5\n3 8\n", "19\n17\n15\n14\n13\n12\n10\n8\n"},
      Node{"example2", "3 3 3 5\n1 10 100\n2 20 200\n1 10 100\n", "400\n310\n310\n301\n301\n"},
      Node{
        "example3",
        "10 10 10 20\n"
        "7467038376 5724769290 292794712 2843504496 3381970101 8402252870 249131806 6310293640 6690322794 6082257488\n"
        "1873977926 2576529623 1144842195 1379118507 6003234687 4925540914 3902539811 3326692703 484657758 2877436338\n"
        "4975681328 8974383988 2882263257 7690203955 514305523 6679823484 4263279310 585966808 3752282379 620585736\n",
        "23379871545\n22444657051\n22302177772\n22095691512\n21667941469\n21366963278\n21287912315\n21279176669\n"
        "21160477018\n21085311041\n21059876163\n21017997739\n20703329561\n20702387965\n20590247696\n20383761436\n"
        "20343962175\n20254073196\n20210218542\n20150096547\n"}
    )
);

namespace {

/// @brief テストケース名出力用
std::ostream& operator<<(std::ostream& os, const Node& n)
{
  return os << n.name;
}
}  // unnamed namespace
