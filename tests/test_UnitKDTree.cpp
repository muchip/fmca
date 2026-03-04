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
#include <iostream>

#include "../FMCA/Clustering"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2
#define NPTS 10000

int main() {
  FMCA::Tictoc T;

  FMCA::Scalar fill_distance = 0;
  FMCA::Scalar separation_radius = FMCA_INF;
  const FMCA::Matrix P = 0.5 * FMCA::Matrix::Random(DIM, NPTS).array() + 0.5;

  std::cout
      << "Cluster splitter:             "
      << FMCA::internal::traits<FMCA::ClusterTree>::Splitter::splitterName()
      << std::endl;
  T.tic();
  FMCA::iVector index_hits(P.cols());
  index_hits.setZero();
  FMCA::UnitKDTree CT(P, 3);
  for (FMCA::Index i = 0; i < P.cols(); ++i) index_hits(CT.indices()[i]) = 1;
  assert(index_hits.sum() == P.cols() && "CT lost indices");

  std::vector<FMCA::Matrix> bbvec;
  for (auto &&it : CT) {
    if (!it.nSons()) {
      for (auto j = 0; j < it.block_size(); ++j)
        assert(FMCA::internal::inBoundingBox(it, P.col(it.indices()[j])) &&
               "point outside leaf bounding box");
      bbvec.push_back(it.bb());
    }
  }
  FMCA::IO::plotBoxes2D("boxes.vtk", bbvec);
  FMCA::Matrix P3(3, P.cols());
  P3.setZero();
  P3.topRows(2) = P;
  FMCA::IO::plotPoints("points.vtk", P3);
  return 0;
}
