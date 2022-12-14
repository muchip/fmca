#include <iostream>

#include "../FMCA/src/util/KMinList.h"

int main() {
  FMCA::KMinList queue(6);
  queue.insert(std::make_pair(10, 0.5));
  queue.insert(std::make_pair(10, 0.25));
  queue.insert(std::make_pair(5, 0.5));
  queue.insert(std::make_pair(5, 0.25));
  queue.insert(std::make_pair(4, 0.25));
  queue.insert(std::make_pair(7, 0.225));

  for (auto &&it : queue.list())
    std::cout << it.second << " " << it.first << std::endl;
  std::cout << std::endl;
  queue.insert(std::make_pair(6, 0.49));
  queue.insert(std::make_pair(7, 0.49));
  for (auto &&it : queue.list())
    std::cout << it.second << " " << it.first << std::endl;

  return 0;
}
