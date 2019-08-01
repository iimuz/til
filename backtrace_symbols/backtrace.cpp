#include <execinfo.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<std::string> getBackTrace_(const int traceSize);
void func1();
void func2();
}  // unnamed namespace

int main() {
  std::cout << "In " << __func__ << "\n";
  func1();
  return EXIT_SUCCESS;
}

namespace {

///
/// @brief バックトレース情報を取得する
/// @param[in] traceSize 取得するトレースサイズの最大数
/// @note backtrace 関数では、トレース情報を取得する。
/// @note backtrace_symbols 関数で、バックトレース情報を文字列に変換できる。
/// @note symbols は解放する必要がある。
///
std::vector<std::string> getBackTrace_(const int traceSize) {
  void* trace[traceSize];
  const int SIZE = backtrace(trace, traceSize);
  std::unique_ptr<char*, decltype(&free)> symbols(
      backtrace_symbols(trace, SIZE), free);

  const std::vector<std::string> RESULTS(symbols.get(), symbols.get() + SIZE);

  return std::move(RESULTS);
}

/// @brief 入れ子呼び出し用関数1
void func1() {
  std::cout << "In " << __func__ << "\n";
  func2();
  std::cout << "Out " << __func__ << "\n";
}

/// @brief 入れ子呼び出し用関数1
void func2() {
  std::cout << "In " << __func__ << "\n";
  const int TRACE_SIZE(10);
  const std::vector<std::string> TRACE_LIST = getBackTrace_(TRACE_SIZE);
  std::cout << "===== trace result =====\n";
  for (const auto& trace : TRACE_LIST) std::cout << trace << "\n";
  std::cout << "==========\n";
  std::cout << "Out " << __func__ << "\n";
}

}  // unnamed namespace
