#include <cstdlib>
#include <sstream>

#include <windows.h>

///
/// @brief エントリポイント
/// @return 正常に終了した場合に 0 を返す。
///
int main() {
  std::stringstream ss("");
  ss << __FILE__ << "(" << __LINE__ << ") " << __FUNCTION__;
  MessageBox(NULL, ss.str().c_str(), "hello", MB_YESNOCANCEL);

  return EXIT_SUCCESS;
}
