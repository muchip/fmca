#include "GenericMatrix.hpp"

int main() {
  GenericMatrix<double> A(5, 5);
  {
    for (auto i = 0; i < 5; ++i)
      for (auto j = 0; j < 5; ++j) A(i, j) = i * j;
    for (auto i = 0; i < 5; ++i) {
      for (auto j = 0; j < 5; ++j) std::cout << A(i, j) << " ";
      std::cout << std::endl;
    }
  }
  GenericMatrix<double> B;
  std::cout << "got here?\n";
  B = A + A;
  std::cout << "got here 2?\n";
  for (auto i = 0; i < 5; ++i) {
    for (auto j = 0; j < 5; ++j) std::cout << B(i, j) << " ";
    std::cout << std::endl;
  }

  return 0;
}
