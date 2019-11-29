/// @file

#define _TEST
#include "../main.cpp"
#undef _TEST

#include "gtest/gtest.h"

#include <sstream>
#include <string>

namespace {

class ExampleTest: public testing::Test {};
}  // unnamed namespace

/// @brief example test
TEST_F(ExampleTest, example)
{
  std::stringstream sin("1 2 3\nbuff\n");
  std::stringstream sout("");
  run(sin, sout);

  const std::string EXP_STR("6 buff\n");
  ASSERT_EQ(EXP_STR, sout.str());
}

