#include<iostream>
#include <FMCA/Clustering>

#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/print2file.hpp"
#include "../FMCA/src/util/tictoc.hpp"
#include "generateSwissCheese.h"
#include "generateSwissCheeseExp.h"

int main() {
  tictoc T;
  unsigned int npts = 1e6;
  T.tic();
  Eigen::MatrixXd P = generateSwissCheeseExp(3, npts);
  //Bembel::IO::print2m("d1c.m", "P", P, "w");
  //return 0;
  T.toc("pts... ");
  Eigen::MatrixXd Q(3, npts);
  //Q.topRows(2) = P;
  Q = P;
  FMCA::ClusterT CT(Q, 10);
  T.tic();
  std::vector<const FMCA::TreeBase<FMCA::ClusterT> *> leafs;
  for (auto level = 0; level < 16; ++level) {
    std::vector<Eigen::MatrixXd> bbvec;
    for (auto &node : CT) {
      if (node.level() == level)
        bbvec.push_back(node.derived().bb());
    }
    FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
  }
  std::vector<Eigen::MatrixXd> bbvec;

  for (auto &node : CT) {
    if (!node.nSons())
      bbvec.push_back(node.derived().bb());
  }
  FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
  FMCA::IO::plotPoints("points.vtk", Q);

  return 0;
}