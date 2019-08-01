#include <iostream>
#include <vector>

namespace {
///
/// @brief テスト用にクラスを生成する。
/// @note PODとかになって最適化されても困るので少し複雑なクラスにしておく。
///
class Sample {
public:
  Sample()                    = default;
  Sample(Sample&& other)      = default;
  Sample(const Sample& other) = default;

  virtual void print() const;

  Sample& operator=(Sample&& rhs) = default;
  Sample& operator=(const Sample& rhs) = default;

private:
  int val;
  std::vector<double> list;
};  // struct Sample

void normalInstance_();
void nullptrInstance_();
void showPointer_(const Sample& obj);
}  // unnamed namespace

///
/// @brief エントリポイント
/// @return 正常に終了した場合に 0 を返す。
int main() {
  normalInstance_();
  nullptrInstance_();

  return EXIT_SUCCESS;
}

namespace {

/// @brief メンバ変数を表示する
void Sample::print() const {
  std::cout << "val = " << this->val;
  std::cout << ", vector size = " << this->list.size();
  std::cout << "\n";
}

/// @brief 通常通りインスタンスを確保した場合の動作
void normalInstance_() {
  Sample instance;
  instance.print();

  std::cout << "original pointer = " << std::hex << &instance << std::dec
            << "\n";
  showPointer_(instance);
}

/// @brief nullptr の実体を渡した場合の動作
void nullptrInstance_() {
  Sample* ptr(nullptr);

  std::cout << "original pointer = " << std::hex << ptr << std::dec << "\n";
  showPointer_(*ptr);
}

/// @brief 実体を渡したときのポインタを出力する。
void showPointer_(const Sample& obj) {
  std::cout << "object pointer = " << std::hex << &obj << std::dec << "\n";
  // obj.print();  // セグメンテーションフォールト
}
}  // namespace
