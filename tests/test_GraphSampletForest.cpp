// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
extern "C" {
#include <metis.h>
}

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <random>
//
#include "../FMCA/Samplets"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

using Graph = FMCA::Graph<idx_t, FMCA::Scalar>;

int main(int argc, char *argv[]) {
  FMCA::Tictoc T;
  // const FMCA::Index npts = 1000000;
  const FMCA::Index k = 100;

  std::mt19937 mt;
  mt.seed(0);
  // FMCA::Matrix P(3, npts);
#if 0
  {
    std::normal_distribution<FMCA::Scalar> dist(0.0, 1.0);
    for (FMCA::Index i = 0; i < P.cols(); ++i) {
      P.col(i) << dist(mt), dist(mt), dist(mt);
      P.col(i) /= P.col(i).norm();
    }
  }
  {
    for (FMCA::Index i = 0; i < npts; ++i) {
      const FMCA::Scalar u = FMCA::Scalar(rand()) / RAND_MAX;
      const FMCA::Scalar v = FMCA::Scalar(rand()) / RAND_MAX;
      const FMCA::Scalar t = 1.5 * FMCA_PI + 4.5 * FMCA_PI * u;
      const FMCA::Scalar x = t * std::cos(t);
      const FMCA::Scalar y = 21.0 * v;
      const FMCA::Scalar z = t * std::sin(t);
      P.col(i) << x, y, z;
    }
  }
#endif
  FMCA::Matrix data =
      FMCA::IO::ascii2Matrix("era5_20-06-24-12h_full.txt").transpose();
  FMCA::Matrix outline =
      FMCA::IO::ascii2Matrix("outline.txt").transpose().topRows(3);
  FMCA::Matrix P = data.topRows(3);
  const FMCA::Index npts = P.cols();
  FMCA::Vector signal(P.cols());
  for (FMCA::Index i = 0; i < P.cols(); ++i)
    signal(i) = P(0, i) * P(0, i) + P(1, i) + P(2, i);
  signal = data.row(3);
  FMCA::IO::plotPointsColor("data.vtk", P, signal);
  FMCA::IO::plotPoints("outline.vtk", outline);

  FMCA::ClusterTree CT(P, 100);
  T.tic();
  std::vector<Eigen::Triplet<FMCA::Scalar>> A = FMCA::symKNN(CT, P, k);
  T.toc("kNN:");
  FMCA::Graph<idx_t, FMCA::Scalar> G;
  G.init(P.cols(), A);
  T.tic();
  FMCA::GraphSampletForest<Graph> gsf(G, 6, 2, 3, 1000);
  T.toc("samplet forest: ");
  for (FMCA::Index i = 0; i < gsf.lost_energies().size(); ++i)
    std::cout << gsf.lost_energies()[i] << std::endl;
  T.tic();
  FMCA::Vector Tsignal = gsf.sampletTransform(signal, 1e-3);
  T.toc("samplet transform: ");
  T.tic();
  FMCA::Vector TtTsignal = gsf.inverseSampletTransform(Tsignal);
  T.toc("inverse samplet transform: ");
  std::cout << "isometry? " << (TtTsignal - signal).norm() / signal.norm()
            << std::endl;
  FMCA::Index nneg = 0;
  FMCA::Scalar nrm = Tsignal.norm();
  for (FMCA::Index i = 0; i < P.cols(); ++i) {
    if (std::abs(Tsignal(i)) > 0) ++nneg;
  }
  FMCA::IO::plotPointsColor("compression_error.vtk", P, signal - TtTsignal);
  std::cout << (signal - TtTsignal).norm() / signal.norm() << std::endl;
  Tsignal.setZero();
  Tsignal(100) = 1;
  TtTsignal = gsf.inverseSampletTransform(Tsignal);
  FMCA::IO::plotPointsColor("samplet.vtk", P, TtTsignal);

  std::cout << "non negligible coeffs: " << nneg << std::endl;

  return 0;
}
