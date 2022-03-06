#include <FMCA/Clustering>
#include <iostream>

#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/print2file.hpp"
#include "../FMCA/src/util/tictoc.hpp"
#include "../Points/matrixReader.h"
#include "generateSwissCheese.h"
#include "generateSwissCheeseExp.h"

int main() {
  tictoc T;
  // unsigned int npts = 1e5;
  T.tic();
  Eigen::MatrixXd P;  // = generateSwissCheeseExp(2, npts);
  Eigen::MatrixXd Pts = readMatrix("../Points/bunnyFine.txt");
  P = Pts.transpose();
  // Bembel::IO::print2m("d1c.m", "P", P, "w");
  // return 0;
  T.toc("pts... ");
  Eigen::MatrixXd Q;
  // Q.setZero();
  // Q.topRows(2) = P;
  Q = P;
  FMCA::ClusterT CT(Q, 10);
  T.tic();
  std::vector<const FMCA::TreeBase<FMCA::ClusterT> *> leafs;
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
