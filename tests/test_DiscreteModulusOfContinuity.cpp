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
#include <random>

#include "../FMCA/Clustering"
#include "../FMCA/src/ModulusOfContinuity/DiscreteModulusOfContinuity.h"
#include "../FMCA/src/ModulusOfContinuity/greedySetCovering.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

namespace FMCA {

// generalization to nD is called Brownian sheet and in its simples version
// just the tensor product of Wiener processes
Vector simulateWienerProcess(const Vector &tvec) {
  Vector B(tvec.size());
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<Scalar> Z(0, 1);

  B(0) = std::sqrt(tvec(0)) * Z(gen);
  for (Index i = 1; i < B.size(); ++i)
    B(i) = B(i - 1) + std::sqrt(tvec(i) - tvec(i - 1)) * Z(gen);

  return B;
}

}  // namespace FMCA

int main() {
  const FMCA::Index nPts = 100000;
  FMCA::Matrix P(1, nPts);
  FMCA::Vector data(nPts);
  P.setRandom();
  FMCA::Scalar h = 1. / nPts;
  for (FMCA::Index i = 0; i < nPts; ++i) {
    P(0, i) = (i + 0.5) * h;
    data(i) = std::sqrt(P(0, i));
  }
  // data = FMCA::simulateWienerProcess(P.row(0));

  FMCA::Tictoc T;
#if 0
  std::mt19937 mt;
  mt.seed(0);
  FMCA::Matrix Psphere(3, nPts);
  FMCA::Matrix Psquare(2, nPts);

  {
    std::normal_distribution<FMCA::Scalar> dist(0.0, 1.0);
    for (FMCA::Index i = 0; i < Psphere.cols(); ++i) {
      Psphere.col(i) << dist(mt), dist(mt), dist(mt);
      Psphere.col(i) /= Psphere.col(i).norm();
    }
    Psquare.setRandom();
  }
#endif
  T.tic();
  FMCA::DiscreteModulusOfContinuity dmoc;
  FMCA::Scalar r = 1e-5;
  FMCA::Index R = 3;
  dmoc.init<FMCA::ClusterTree>(P, data, r, R, 0.1, 1, "EUCLIDEAN");
  T.toc("set up dmoc: ");
  const std::vector<std::vector<FMCA::Index>> &idcs = dmoc.XNk_indices();
  for (FMCA::Index i = 0; i < dmoc.omegaNk().size(); ++i) {
    FMCA::Matrix Ploc(3, idcs[i].size());
    Ploc.setZero();
    std::cout << dmoc.omegaNk()[i] << " " << dmoc.omegat()[i] << std::endl;
    for (FMCA::Index j = 0; j < Ploc.cols(); ++j)
      Ploc.col(j) << P(0, idcs[i][j]), data(idcs[i][j]), 0;
    FMCA::IO::plotPointsColor("sub" + std::to_string(i) + ".vtk", Ploc,
                              Ploc.row(1).transpose());
  }
  std::printf("t            omega(t)\n");
  for (int i = 6; i >= 0; --i) {
    const FMCA::Scalar t = std::pow(10., -i);
    std::printf("%5.2e     %10.6f\n", t,
                dmoc.omega<FMCA::ClusterTree>(t, P, data));
  }
  return 0;
}
