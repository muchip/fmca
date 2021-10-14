#include "FMCA/src/util/TreeBase.h"

struct Tree {
  typedef int NodeData;
};
int main() {
  FMCA::TreeBase<Tree> myTree;

  return 0;
}
