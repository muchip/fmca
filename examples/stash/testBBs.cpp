#include <FMCA/Clustering>
#include <iostream>

#include "../../FMCA/src/util/IO.h"
#include "../../FMCA/src/util/print2file.h"
#include "../../Points/matrixReader.h"
#include "generateSwissCheese.h"
#include "generateSwissCheeseExp.h"

int main() {
  Eigen::MatrixXd P(2, 1000);  // = generateSwissCheeseExp(2, npts);
  for (auto i = 0; i < 1000; ++i) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(2);
    while (x.norm() > 1 || x.norm() < 0.5) x = Eigen::VectorXd::Random(2);
    P.col(i) = x;
  }

  Eigen::MatrixXd Q(3, 1000);
  Q.setZero();
  Q.topRows(2) = P;
  FMCA::ClusterTree CT(Q, 4);
  std::vector<const FMCA::TreeBase<FMCA::ClusterTree> *> leafs;
  for (auto level = 0; level < 16; ++level) {
    std::vector<Eigen::MatrixXd> bbvec;
    for (auto &node : CT) {
      if (node.level() == level) bbvec.push_back(node.derived().bb());
    }
    FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
  }
  std::vector<Eigen::MatrixXd> bbvec;

  for (auto &node : CT) {
    if (!node.nSons()) bbvec.push_back(node.derived().bb());
  }
  Eigen::VectorXd colrs(P.cols());
  for (auto &node : CT) {
    if (!node.nSons())
      for (auto it = node.indices().begin(); it != node.indices().end(); ++it)
        colrs(*it) = node.block_id();
  }
  FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
  FMCA::IO::plotPointsColor("points.vtk", Q, colrs);

  return 0;
}
