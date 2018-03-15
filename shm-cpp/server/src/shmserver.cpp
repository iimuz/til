#include <cstdlib>
#include <iostream>

#include <boost/interprocess/shared_memory_object.hpp>

/// @brief IPC用の共有メモリを提供するサーバ側のエントリポイント
int main() {
  std::cout << "start hared memory server\n";

  boost::interprocess::shared_memory_object shmObj(boost::interprocess::create_only, "shared_memory", boost::interprocess::read_write);

  std::cout << "end shared memory server\n";
  return EXIT_SUCCESS;
}
