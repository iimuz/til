#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

bool putsNullStr_();
bool writeNullBinary_();
}  // unnamed namespace

///
/// @brief エントリポイント
/// @return 正常に終了した場合に 0 を返す。
///
int main() {
  try {
    if (putsNullStr_() == false) {
      std::cerr << "error has occured in putsNullStr_.\n";
      return EXIT_FAILURE;
    }
    if (writeNullBinary_() == false) {
      std::cerr << "error has occured in writeNullBinary_.\n";
      return EXIT_FAILURE;
    }
  } catch (const std::exception& e) {
    std::cerr << "catch exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

namespace {

///
/// @brief null 終端で埋められた文字列を出力する
///
/// @note null 終端で埋められているので、文字が出力できないはず。
///
bool putsNullStr_() {
  using buff_t = char;

  const std::size_t BUFF_SIZE(100);
  const std::unique_ptr<buff_t[]> buff(new buff_t[BUFF_SIZE]);
  std::fill(buff.get(), buff.get() + BUFF_SIZE, 0U);

  const std::string FILEPATH("../_data/fputs_null.txt");
  const std::unique_ptr<FILE, decltype(&fclose)> fp(
      fopen(FILEPATH.c_str(), "wb"), fclose);
  if (fp.get() == nullptr) return false;
  if (fputs(buff.get(), fp.get()) < 1) return false;

  return true;
}

/// @brief 作りたいパターンの null 終端で埋められたデータの作成
bool writeNullBinary_() {
  using buff_t = char;

  const std::size_t BUFF_SIZE(100);
  const std::unique_ptr<buff_t[]> buff(new buff_t[BUFF_SIZE]);
  std::fill(buff.get(), buff.get() + BUFF_SIZE, 0);

  const std::string FILEPATH("../_data/fputs_null_write.txt");
  std::ofstream ofs(FILEPATH.c_str(), std::ios::binary);
  if (ofs.fail()) return false;
  ofs.write(reinterpret_cast<const char*>(buff.get()),
            sizeof(buff_t) * BUFF_SIZE);
  if (ofs.fail()) return false;

  return true;
}
}  // namespace
