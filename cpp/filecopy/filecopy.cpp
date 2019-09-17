#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

/// @brief 処理の終了状態を表すコード
enum RESULT_CODE {
  NONE = -1,

  SUCCESS,
  FAILURE,

  RESULT_CODE_NUM
};  // enum RESULT_CODE

RESULT_CODE copyByFILE_(const std::string& src, const std::string& dst,
                        const std::size_t bufferByteSize);
RESULT_CODE copyByFstream_(const std::string& src, const std::string& dst,
                           const std::size_t bufferByteSize);
RESULT_CODE copyByRdbuf_(const std::string& src, const std::string& dst);
RESULT_CODE copyByStreamBuf_(const std::string& src, const std::string& dst);
RESULT_CODE createRandomText_(const std::string& filename,
                              const std::size_t byteSize);
RESULT_CODE processTime_(
    const std::string& prefix, const std::string& src, const std::string& dst,
    const std::size_t count,
    const std::function<RESULT_CODE(const std::string&, const std::string&)>
        func);
}  // namespace

///
/// @brief エントリポイント
/// @return 正常終了の場合に EXIT_SUCCESS を返す。
/// @retval EXIT_SUCCESS 正常終了
/// @retval EXIT_FAILURE 異常終了
///
int main() {
  const std::size_t FILESIZE_BYTE(8 * 1024 * 1024);
  const std::string SRC_FILENAME("../_data/filecopy_src.txt");
  const std::string DST_FILENAME("../_data/filecopy_dst.txt");
  const std::size_t MEASURE_COUNT(10);

  // コピー用のファイルを生成
  if (createRandomText_(SRC_FILENAME, FILESIZE_BYTE) != SUCCESS) {
    std::cerr << "create source file error: " << SRC_FILENAME << "\n";
    return EXIT_FAILURE;
  }

  // 各手法の実行
  if (processTime_("stream buf", SRC_FILENAME, DST_FILENAME, MEASURE_COUNT,
                   copyByStreamBuf_) != SUCCESS) {
    return EXIT_FAILURE;
  }
  if (processTime_("rdbuf", SRC_FILENAME, DST_FILENAME, MEASURE_COUNT,
                   copyByRdbuf_) != SUCCESS) {
    return EXIT_FAILURE;
  }
  if (processTime_("fstream[1kB]", SRC_FILENAME, DST_FILENAME, MEASURE_COUNT,
                   [](const std::string& src, const std::string& dst) {
                     return copyByFstream_(src, dst, 1024);
                   }) != SUCCESS) {
    return EXIT_FAILURE;
  }
  if (processTime_("fstream[256kB]", SRC_FILENAME, DST_FILENAME, MEASURE_COUNT,
                   [](const std::string& src, const std::string& dst) {
                     return copyByFstream_(src, dst, 256 * 1024);
                   }) != SUCCESS) {
    return EXIT_FAILURE;
  }
  if (processTime_("file[1kB]", SRC_FILENAME, DST_FILENAME, MEASURE_COUNT,
                   [](const std::string& src, const std::string& dst) {
                     return copyByFILE_(src, dst, 1024);
                   }) != SUCCESS) {
    return EXIT_FAILURE;
  }
  if (processTime_("file[256kB]", SRC_FILENAME, DST_FILENAME, MEASURE_COUNT,
                   [](const std::string& src, const std::string& dst) {
                     return copyByFILE_(src, dst, 256 * 1024);
                   }) != SUCCESS) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

namespace {

/// @brief 時間計測用クラス
class Timer {
public:
  Timer() : m_start(clock_t::now()){};
  Timer(Timer&& other)      = default;
  Timer(const Timer& other) = default;
  ~Timer()                  = default;

  void reset() { this->m_start = clock_t::now(); }
  double elapsed() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               clock_t::now() - this->m_start)
        .count();
  }

  Timer& operator=(Timer& rhs) = default;
  Timer& operator=(const Timer& rhs) = default;

private:
  using clock_t = std::chrono::high_resolution_clock;

  std::chrono::time_point<clock_t> m_start;
};  // class Timer

///
/// @brief C の FILE を利用したファイルコピー
/// @param[in] src 元ファイル
/// @param[in] dst 出力ファイル
/// @param[in] bufferByteSize バッファサイズ
/// @return RESULT_CODE
///
/// @note 指定したバッファサイズ単位で読み込みと書き出しを行う。
///
RESULT_CODE copyByFILE_(const std::string& src, const std::string& dst,
                        const std::size_t bufferByteSize) {
  if (bufferByteSize <= 0) return FAILURE;

  std::unique_ptr<FILE, decltype(&fclose)> ifp(fopen(src.c_str(), "rb"),
                                               fclose);
  std::unique_ptr<FILE, decltype(&fclose)> ofp(fopen(dst.c_str(), "wb"),
                                               fclose);
  if ((ifp == nullptr) || (ofp == nullptr)) return FAILURE;

  const std::unique_ptr<char[]> buffer(new char[bufferByteSize]);
  while (feof(ifp.get()) == false) {
    const std::size_t READ_SIZE =
        fread(buffer.get(), 1, bufferByteSize, ifp.get());
    fwrite(buffer.get(), 1, READ_SIZE, ofp.get());
  }

  return SUCCESS;
}

///
/// @brief fstream を利用したファイルコピー
/// @param[in] src 元ファイル
/// @param[in] dst 出力ファイル
/// @param[in] bufferByteSize バッファサイズ
/// @return RESULT_CODE
///
/// @note 指定したバッファサイズ単位で読み込みと書き出しを実行する。
///
RESULT_CODE copyByFstream_(const std::string& src, const std::string& dst,
                           const std::size_t bufferByteSize) {
  if (bufferByteSize <= 0) return FAILURE;

  std::ifstream ifs(src, std::ios::binary);
  std::ofstream ofs(dst, std::ios::binary);
  if (ifs.fail() || ofs.fail()) return FAILURE;

  const std::unique_ptr<char[]> buffer(new char[bufferByteSize]);
  while (true) {
    const std::size_t READ_SIZE =
        ifs.read(buffer.get(), bufferByteSize).gcount();
    if (ifs.eof()) break;

    ofs.write(buffer.get(), READ_SIZE);
  };

  return SUCCESS;
}

///
/// @brief rdbuf を利用したファイルコピー
/// @param[in] src 元ファイル
/// @param[in] dst 出力ファイル
/// @return RESULT_CODE
///
/// @note 一括で読み込みが発生するため、巨大ファイルに適さない。
///
RESULT_CODE copyByRdbuf_(const std::string& src, const std::string& dst) {
  std::ifstream ifs(src, std::ios::binary);
  std::ofstream ofs(dst, std::ios::binary);
  if (ifs.fail() || ofs.fail()) return FAILURE;

  ofs << ifs.rdbuf();
  if (ifs.fail() || ofs.fail()) return FAILURE;

  return SUCCESS;
}

///
/// @brief streambuf_iterator を利用したファイルコピー
/// @param[in] src 元ファイル
/// @param[in] dst 出力ファイル
/// @return RESULT_CODE
///
/// @note 1 byte ずつ読み込みと書き出しが発生するため遅くなる。
///
RESULT_CODE copyByStreamBuf_(const std::string& src, const std::string& dst) {
  std::ifstream ifs(src, std::ios::binary);
  std::ofstream ofs(dst, std::ios::binary);
  if (ifs.fail() || ofs.fail()) return FAILURE;

  std::istreambuf_iterator<char> srcIt(ifs);
  std::istreambuf_iterator<char> end;
  std::ostreambuf_iterator<char> dstIt(ofs);
  std::copy(srcIt, end, dstIt);
  if (ifs.fail() || ofs.fail()) return FAILURE;

  return SUCCESS;
}

///
/// @brief 指定したサイズのファイルを生成する
/// @param[in] filename ファイル名
/// @param[in] byteSize 作成するサイズ
///
RESULT_CODE
createRandomText_(const std::string& filename, const std::size_t byteSize) {
  if (byteSize <= 0) return FAILURE;

  std::ofstream ofs(filename, std::ios::binary);
  if (ofs.fail()) return FAILURE;

  const std::size_t BUFFER_SIZE_BYTE(64 * 1024 * 1024);
  const std::unique_ptr<char[]> buffer(new char[BUFFER_SIZE_BYTE]);
  std::memset(buffer.get(), 'a', BUFFER_SIZE_BYTE);

  const std::size_t BUFFER_LOOP(byteSize / BUFFER_SIZE_BYTE);
  const std::size_t BUFFER_MOD(byteSize % BUFFER_SIZE_BYTE);
  for (std::size_t i = 0; i < BUFFER_LOOP; ++i) {
    ofs.write(buffer.get(), BUFFER_SIZE_BYTE);
  }
  ofs.write(buffer.get(), BUFFER_MOD);

  return SUCCESS;
}

/// @brief 時間計測を実行する
RESULT_CODE processTime_(
    const std::string& prefix, const std::string& src, const std::string& dst,
    const std::size_t count,
    const std::function<RESULT_CODE(const std::string&, const std::string&)>
        func) {
  std::cout << "=== " << prefix << " ===\n";

  if (count <= 0) return FAILURE;

  std::vector<double> timeList;
  for (std::size_t i = 0; i < count; ++i) {
    const Timer TIMER;
    const RESULT_CODE RESULT = func(src, dst);
    if (RESULT != SUCCESS) {
      std::cerr << prefix << "error."
                << "src: " << src << ", dst: " << dst << "\n";
      return FAILURE;
    }
    timeList.emplace_back(TIMER.elapsed());
  }

  const auto MIN_MAX_ITER =
      std::minmax_element(timeList.cbegin(), timeList.cend());
  const double SUM_VAL =
      std::accumulate(timeList.cbegin(), timeList.cend(), 0.);
  const double MEAN_VAL = SUM_VAL / count;

  std::cout << "min: " << *MIN_MAX_ITER.first
            << " [msec]"
               ", max: "
            << *MIN_MAX_ITER.second
            << " [msec]"
               ", mean: "
            << MEAN_VAL << " [msec]\n";
  return SUCCESS;
}  // namespace
}  // unnamed namespace
