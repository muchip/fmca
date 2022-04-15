#include "../FMCA/src/util/HaltonSet.h"
#include <iostream>
int main() {

  FMCA::HaltonSet<1000> HS(4);
  for (auto &&it : HS.primes())
    std::cout << it << " ";
  std::cout << std::endl;
  for (auto i = 0; i < 10; ++i) {
    std::cout << HS.EigenHaltonVector().transpose() << std::endl;
    HS.next();
  }
  return 0;
}
