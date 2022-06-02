// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#include <Eigen/Dense>
#include <FMCA/Clustering>
#include <iomanip>
#include <iostream>
//
#include <FMCA/src/util/IO.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>

#include "../Points/matrixReader.h"

int main() {
  // const FMCA::Matrix Q = readMatrix("../Points/enterpriseD.txt");
  // const FMCA::Matrix P = Q.transpose();
  const FMCA::Matrix P = FMCA::Matrix::Random(5, 1000000);
  const FMCA::Index d = FMCA::Index(P.rows());
  const FMCA::Index N = FMCA::Index(P.cols());

  FMCA::Tictoc T;

  T.tic();
  FMCA::ClusterTree CT(P, 10);
  T.toc("cluster tree assembly time:");

  FMCA::clusterTreeStatistics(CT, P);
#ifdef FMCA_PLOT_BOXES_
  std::vector<const FMCA::TreeBase<FMCA::ClusterTree> *> leafs;
  for (auto level = 0; level < 16; ++level) {
    std::vector<Eigen::MatrixXd> bbvec;
    for (auto &node : CT) {
      if (node.level() == level) bbvec.push_back(node.derived().bb());
    }
    FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
  }
  std::vector<FMCA::Matrix> bbvec;

  for (auto &node : CT) {
    if (!node.nSons()) bbvec.push_back(node.derived().bb());
  }
  FMCA::Vector colrs(P.cols());
  for (auto &node : CT) {
    if (!node.nSons()) {
      FMCA::Index idx = rand() % 512;
      for (auto it = node.indices().begin(); it != node.indices().end(); ++it)
        colrs(*it) = idx;
    }
  }
  FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
  FMCA::IO::plotPointsColor("points.vtk", P, colrs);
#endif
  return 0;
}
