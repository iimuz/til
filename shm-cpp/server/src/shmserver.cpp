#include <cstdlib>
#include <iostream>
#include <thread>

#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

namespace bipc = boost::interprocess;

/// @brief IPC用の共有メモリを提供するサーバ側のエントリポイント
int main() {
  std::cout << "start shared memory server\n";

  // 共有メモリの確保
  const int MEM_SIZE(1024);
  bipc::shared_memory_object shmObj(bipc::open_or_create, "shared_memory",
                                    bipc::read_write);
  shmObj.truncate(MEM_SIZE);

  // 共有メモリ情報の出力
  std::cout << "shared memory name: " << shmObj.get_name() << "\n";
  bipc::offset_t size(0);
  if (shmObj.get_size(size)) {
    std::cout << "shared memory size: " << size << "\n";
  }

  // 共有メモリの先頭部分のみ一定間隔で書き換え
  bipc::mapped_region region(shmObj, bipc::read_write);
  int* i1 = static_cast<int*>(region.get_address());
  const int LOOP_NUM(10);
  const auto DULATION = std::chrono::seconds(1);
  for (int i = 0; i < LOOP_NUM; ++i) {
    *i1 = i;
    std::cout << "write: " << i << "\n";
    std::this_thread::sleep_for(DULATION);
  }

  std::cout << "end shared memory server\n";
  return EXIT_SUCCESS;
}
