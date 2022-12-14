// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "../FMCA/Clustering"
#include "IO.h"

#define DIM 3
#define NPTS 10
#define KMINS 2

int main() {
  const FMCA::Index leaf_size = 1;
  FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  {
    FMCA::Matrix Q = FMCA::Matrix::Random(DIM, NPTS);
    Q.row(2) *= 0.;
    FMCA::Index k = 0;
    while (k < NPTS / 3) {
      FMCA::Matrix rdm = FMCA::Matrix::Random(DIM, 1);
      if (rdm.norm() < 1) {
        Q.col(NPTS / 3 + k) = rdm + 1.5 * FMCA::Matrix::Ones(DIM, 1);
        ++k;
      }
    }
    FMCA::Matrix dir(DIM, 1);
    dir.col(0) << -0.5, 0.5, 1.;
    Q.rightCols(NPTS / 3) = dir * (FMCA::Matrix::Random(1, NPTS / 3) +
                                   FMCA::Matrix::Ones(1, NPTS / 3));
    Q.rightCols(NPTS / 3).row(2).array() += 0.5;
    std::vector<int> subsample(NPTS);
    std::iota(subsample.begin(), subsample.end(), 0);
    std::random_shuffle(subsample.begin(), subsample.end());
    for (auto i = 0; i < NPTS; ++i) P.col(i) = Q.col(subsample[i]);
#if 0
    std::ofstream myfile;
    myfile.open("pts.txt");
    for (auto i = 0; i < P.cols(); ++i)
      myfile << P(0, i) << " "
             << " " << P(1, i) << " " << P(2, i) << std::endl;
    myfile.close();
#endif
  }

  FMCA::ClusterTree CT(P, leaf_size);
  FMCA::iMatrix kMins = kMinDistance(CT, P, KMINS);
  std::cout << kMins << std::endl;
  auto idcs = CT.indices();
  for (auto i = 0; i < P.cols(); ++i) {
    for (auto j = 0; j < KMINS; ++j)
      std::cout << (P.col(i) - P.col(kMins(i, j))).norm() << " ";
    std::cout << std::endl;
  }
  FMCA::Matrix distMat(P.cols(), P.cols());
  for (auto j = 0; j < P.cols(); ++j)
    for (auto i = 0; i < P.cols(); ++i)
      distMat(i, j) = (P.col(i) - P.col(j)).norm();
  std::cout << distMat << std::endl;
#if 0
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
  FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
  FMCA::Vector colrs(P.cols());
  for (auto &node : CT) {
    if (!node.nSons()) {
      FMCA::Index idx = rand() % 512;
      for (auto it = node.indices().begin(); it != node.indices().end(); ++it)
        colrs(*it) = idx;
    }
  }
  FMCA::IO::plotPointsColor("points.vtk", P, colrs);
#endif
  return 0;
}
