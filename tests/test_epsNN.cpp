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
#include "../FMCA/src/Clustering/epsNN.h"
#include "../FMCA/src/util/Tictoc.h"

template <typename Dists>
struct my_less {
  my_less(const Dists &dist) : dist_(dist) {}
  template <typename idx>
  bool operator()(const idx &a, const idx &b) const {
    return dist_(a) < dist_(b);
  }
  const Dists &dist_;
};

#define DIM 50

int main(int argc, char *argv[]) {
  FMCA::Tictoc T;
  const FMCA::Index npts = 50000;
  const FMCA::Index leaf_size = 100;
  const FMCA::Scalar eps = 0.8;

  FMCA::Matrix P = FMCA::Matrix::Random(DIM, npts);
  FMCA::Matrix pts = FMCA::Matrix::Random(DIM, 10);
  T.tic();
  FMCA::ClusterTree CT(P, leaf_size);
  T.toc("cluster tree: ");
  T.tic();
  const FMCA::Vector mdv = FMCA::minDistanceVector(CT, P);
  T.toc("min dist: ");
  std::cout << mdv.minCoeff() << " " << mdv.maxCoeff() << std::endl;
  for (FMCA::Index i = 0; i < pts.cols(); ++i) {
    T.tic();
    std::vector<FMCA::Index> idcs = FMCA::epsNN(CT, P, pts.col(i), eps);
    T.toc("time epsNN");
    for (FMCA::Index j = 0; j < idcs.size(); ++j)
      assert((P.col(idcs[j]) - pts.col(i)).norm() < eps && "false index added");
    FMCA::Index cnt = 0;
    for (FMCA::Index j = 0; j < P.cols(); ++j)
      if ((P.col(j) - pts.col(i)).norm() < eps) ++cnt;
    assert(cnt == idcs.size() && "there are missing indices");
    std::cout << idcs.size() << std::endl;
  }
  return 0;
}
