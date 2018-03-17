#include <cstdlib>
#include <iostream>
#include <thread>

#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

namespace bipc = boost::interprocess;

/// @brief 共有メモリを利用したIPCのクライアントのエントリポイント
int main() {
  std::cout << "start shared memory client.\n";

  // 共有メモリへのアクセス
  bipc::shared_memory_object shmObj(bipc::open_only, "shared_memory",
                                    bipc::read_only);

  // 共有メモリ情報の出力
  std::cout << "shared memory name: " << shmObj.get_name() << "\n";
  bipc::offset_t size(0);
  if (shmObj.get_size(size)) {
    std::cout << "shared memory size: " << size << "\n";
  }

  // 共有メモリの内容を書き出し
  const int LOOP_NUM(10);
  const auto DULATION(std::chrono::seconds(1));
  bipc::mapped_region region(shmObj, bipc::read_only);
  const int* i1 = static_cast<int*>(region.get_address());
  for (int i = 0; i < LOOP_NUM; ++i) {
    std::cout << "read: " << *i1 << "\n";
    std::this_thread::sleep_for(DULATION);
  }

  std::cout << "end shared memory client.\n";
  return EXIT_SUCCESS;
}
