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

#define DIM 3
#define NPTS 1000

int main() {
  FMCA::Tictoc T;
  FMCA::Scalar fill_distance = 0;
  FMCA::Scalar separation_radius = 1. / 0.;
  FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, 8 * 8 * 8);
  FMCA::iVector iidcs = FMCA::iVector::LinSpaced(8, 0, 7);
  FMCA::Index l = 0;
  for (FMCA::Index i = 0; i < 8; ++i) {
    for (FMCA::Index j = 0; j < 8; ++j) {
      for (FMCA::Index k = 0; k < 8; ++k) {
        P.col(l) << iidcs(i), iidcs(j), iidcs(k);
        ++l;
      }
    }
  }
  P = P + 0.1 * Eigen::MatrixXd::Random(DIM, 8 * 8 * 8);
  std::cout
      << "Cluster splitter:             "
      << FMCA::internal::traits<FMCA::ClusterTree>::Splitter::splitterName()
      << std::endl;
  FMCA::MortonClusterTree MCT(P, 10);
  FMCA::ClusterTree CT(P, 10);
  clusterTreeStatistics(CT, P);
  clusterTreeStatistics(MCT, P);
  std::vector<FMCA::Matrix> bbvec;
  FMCA::Vector colr(P.cols());
  for (auto &&it : MCT) {
    if (!it.nSons()) {
      double rdm = rand() % P.cols();
      for (auto j = 0; j < it.block_size(); ++j) {
        assert(FMCA::internal::inBoundingBox(it, P.col(it.indices()[j])) &&
               "point outside leaf bounding box");
        colr(it.indices()[j]) = it.block_id();
      }
      bbvec.push_back(it.bb());
    }
  }
  FMCA::IO::plotBoxes("boxesMorton.vtk", bbvec);
  FMCA::IO::plotPointsColor("points.vtk", P, colr);
  bbvec.clear();
  for (auto &&it : CT) {
    if (!it.nSons()) {
      for (auto j = 0; j < it.block_size(); ++j)
        assert(FMCA::internal::inBoundingBox(it, P.col(it.indices()[j])) &&
               "point outside leaf bounding box");
      bbvec.push_back(it.bb());
    }
  }
  FMCA::IO::plotBoxes("boxes.vtk", bbvec);
  FMCA::iVector idcs =
      Eigen::Map<const FMCA::iVector>(MCT.indices(), MCT.block_size());
  FMCA::IO::print2m("output.m", "P", P, "w");
  FMCA::IO::print2m("output.m", "I", idcs, "a");
  return 0;
}
