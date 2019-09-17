#include <cstdlib>
#include <iostream>

namespace {

void segFaultFunction_();
}  // unnamed namespace

int main() {
  std::cout << "In " << __func__ << "\n";
  segFaultFunction_();
  return EXIT_SUCCESS;
}

namespace {

/// @brief 強制的に Segmentation Fault を発生させる関数
void segFaultFunction_() {
  std::cout << "In " << __func__ << "\n";
  int* p(nullptr);
  *p = 10;
  std::cout << "Out " << __func__ << "\n";
}
}  // unnamed namespace
