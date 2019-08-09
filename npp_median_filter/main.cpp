#include <chrono>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <npp.h>
#include <nppi.h>

namespace {

long long getMicroSec_(const std::chrono::system_clock::time_point& start,
                       const std::chrono::system_clock::time_point& end);
template <typename T>
std::unique_ptr<T, decltype(&cudaFree)> cudaMalloc_(const std::size_t elemNum,
                                                    cudaError_t& err);
}  // unnamed namespace

int main() {
  using val_t       = Npp16s;
  using image_t     = std::unique_ptr<val_t, decltype(&cudaFree)>;
  using bufferVal_t = Npp8u;
  using buffer_t    = std::unique_ptr<bufferVal_t, decltype(&cudaFree)>;

  const int WIDTH(2048);
  const int HEIGHT(1024);
  const int SIZE(WIDTH * HEIGHT);

  // 計算用メモリ確保
  image_t src(nullptr, &cudaFree);
  image_t dst(nullptr, &cudaFree);
  {
    cudaDeviceSynchronize();
    const auto START_TIME = std::chrono::system_clock::now();
    cudaError_t err(cudaSuccess);

    src = cudaMalloc_<val_t>(SIZE, err);
    cudaMemset(src.get(), 0, sizeof(val_t) * SIZE);

    dst = cudaMalloc_<val_t>(SIZE, err);
    cudaMemset(src.get(), 0, sizeof(val_t) * SIZE);

    cudaDeviceSynchronize();
    const auto END_TIME = std::chrono::system_clock::now();
    std::cout << "cuda malloc time: "
              << 1e-3 * getMicroSec_(START_TIME, END_TIME) << " [ms]"
              << "\n";
    const auto ERR = cudaGetLastError();
    if (ERR != cudaSuccess) {
      std::cout << "cuda error: " << ERR << "\n";
      return EXIT_FAILURE;
    }
  }

  const Npp32s STEP(WIDTH * sizeof(val_t));
  const NppiSize ROI{WIDTH, HEIGHT};
  const NppiSize MASK{5, 5};
  const NppiPoint ANCHOR{0, 0};

  // 計算用バッファの取得
  buffer_t buffer(nullptr, &cudaFree);
  {
    cudaDeviceSynchronize();
    const auto START_TIME = std::chrono::system_clock::now();
    Npp32u bufferSize(0);
    const auto STATUS =
        nppiFilterMedianGetBufferSize_16s_C1R(ROI, MASK, &bufferSize);
    cudaDeviceSynchronize();
    if (STATUS != NPP_SUCCESS) {
      std::cout << "npp error: " << STATUS << "\n";
      return EXIT_FAILURE;
    }

    cudaError_t err;
    buffer = cudaMalloc_<bufferVal_t>(SIZE, err);

    cudaDeviceSynchronize();
    const auto END_TIME = std::chrono::system_clock::now();
    std::cout << "prepare median buffer size: "
              << 1e-3 * getMicroSec_(START_TIME, END_TIME) << " [ms]"
              << "\n";

    const auto ERR = cudaGetLastError();
    if (ERR != cudaSuccess) {
      std::cout << "cuda error: " << ERR << "\n";
      return EXIT_FAILURE;
    }
  }

  // メディアンフィルタを規定回数繰り返す
  const int LOOP_MAX(10);
  for (int i = 0; i < 10; ++i) {
    cudaDeviceSynchronize();
    const auto START_TIME = std::chrono::system_clock::now();
    const auto STATUS     = nppiFilterMedian_16s_C1R(
        src.get(), STEP, dst.get(), STEP, ROI, MASK, ANCHOR, buffer.get());
    cudaDeviceSynchronize();
    const auto END_TIME = std::chrono::system_clock::now();
    std::cout << "median filter time [" << i << "/ " << LOOP_MAX
              << "]: " << 1e-3 * getMicroSec_(START_TIME, END_TIME) << " [ms]"
              << "\n";

    if (STATUS != NPP_SUCCESS) {
      std::cout << "npp error: " << STATUS << "\n";
      return EXIT_FAILURE;
    }
    const auto ERR = cudaGetLastError();
    if (ERR != cudaSuccess) {
      std::cout << "cuda error: " << ERR << "\n";
      return EXIT_FAILURE;
    }
  }

  // メモリ解放処理
  {
    cudaDeviceSynchronize();
    const auto START_TIME = std::chrono::system_clock::now();
    src.reset(nullptr);
    dst.reset(nullptr);
    buffer.reset(nullptr);
    cudaDeviceSynchronize();
    const auto END_TIME = std::chrono::system_clock::now();
    std::cout << "delete buffer: " << 1e-3 * getMicroSec_(START_TIME, END_TIME)
              << " [ms]"
              << "\n";
  }

  return EXIT_SUCCESS;
}

namespace {

/// @biref 差分を micro sec 単位で取得する
long long getMicroSec_(const std::chrono::system_clock::time_point& start,
                       const std::chrono::system_clock::time_point& end) {
  const auto diff =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  return diff.count();
}

///
/// @brief 指定した形式の cuda メモリを確保する
/// @param[in] elemNum 確保する要素数
/// @param[out] err 実行した結果の cudaError_t
/// @return 確保したメモリ
/// @note elemNum * sizeof(T) のメモリを確保する
///
template <typename T>
std::unique_ptr<T, decltype(&cudaFree)> cudaMalloc_(const std::size_t elemNum,
                                                    cudaError_t& err) {
  T* p(nullptr);
  err = cudaMalloc(reinterpret_cast<void**>(&p), sizeof(T) * elemNum);
  return std::unique_ptr<T, decltype(&cudaFree)>(p, cudaFree);
}
}  // unnamed namespace
