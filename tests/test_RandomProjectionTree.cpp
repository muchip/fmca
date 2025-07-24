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
#define NPTS 1000000

int main() {
  FMCA::Tictoc T;
  FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, NPTS);
  FMCA::Vector colr(NPTS);
  std::vector<FMCA::Scalar> colr2;
  T.tic();
  FMCA::RandomProjectionTree ct(P, 100);
  T.toc("tree computation: ");
  std::vector<FMCA::Matrix> bbvec;
  for (const auto &it : ct) {
    if (!it.nSons()) {
      const FMCA::Index rdm = std::rand() % 256;
      for (FMCA::Index i = 0; i < it.block_size(); ++i) {
        colr(it.indices()[i]) = rdm;
      }
      if (it.block_size()) {
        bbvec.push_back(it.bb());
        colr2.push_back(rdm);
      }
    }
  }
  FMCA::clusterTreeStatistics(ct, P);
  std::vector<FMCA::Index> found(NPTS);
  for (FMCA::Index i = 0; i < ct.block_size(); ++i) ++(found[ct.indices()[i]]);
  for (FMCA::Index i = 0; i < found.size(); ++i)
    assert(found[i] == 1 && "index mismatch");
  FMCA::Matrix P3D(3, NPTS);
  P3D.setZero();
  P3D.topRows(2) = P;
  FMCA::IO::plotPointsColor("clusters.vtk", P3D, colr);
  FMCA::IO::plotBoxes2D("boxes.vtk", bbvec, colr2);

  std::cout << "testing fill distance and separation radius\n";
  {
    FMCA::Matrix P = Eigen::MatrixXd::Random(20, 100000);
    FMCA::ClusterTree ct(P, 100);
    T.tic();
    FMCA::Vector ex_min_dist = FMCA::minDistanceVector(ct, P);
    T.toc("exact min distance: ");
    T.tic();
    FMCA::Vector min_dist = FMCA::fastMinDistanceVector(P);
    T.toc("app min distance: ");
    std::cout << "sep / fill exact: " << ex_min_dist.minCoeff() << " "
              << ex_min_dist.maxCoeff() << std::endl;
    std::cout << "sep / fill approx: " << min_dist.minCoeff() << " "
              << min_dist.maxCoeff() << std::endl;
    std::cout << "err: "
              << std::abs(ex_min_dist.minCoeff() - min_dist.minCoeff()) << " "
              << std::abs(ex_min_dist.maxCoeff() - min_dist.maxCoeff())
              << std::endl;
  }
  return 0;
}
