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
#include <FMCA/src/util/IO.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/uniformSphericalPoints.h>

#include <FMCA/H2Matrix>
#include <FMCA/HMatrix>
#include <FMCA/Kernel>
#include <iostream>

#define NPTS 10000
#define DIM 3
#define MPOLE_DEG 5

using SphericalKernel = FMCA::PDKernel<FMCA::Metric::GeodesicSphere>;
using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, SphericalKernel>;
using MatrixEvaluatorUS =
    FMCA::unsymmetricNystromEvaluator<Moments, SphericalKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::SphereClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree, FMCA::CompareSphericalCluster>;
using HMatrix =
    FMCA::HMatrix<FMCA::SphereClusterTree, FMCA::CompareSphericalCluster>;

int main() {
  FMCA::Tictoc T;
  SphericalKernel function("EXPONENTIAL", 2.);
  const FMCA::Matrix P = FMCA::uniformSphericalPoints(NPTS, 0);
  FMCA::Vector col0 = function.eval(P, P.col(0));

  FMCA::IO::plotPointsColor("sig2.vtk", P, col0);
  const Moments mom(P, MPOLE_DEG);

  std::cout << "Cluster splitter:             "
            << FMCA::internal::traits<
                   FMCA::SphereClusterTree>::Splitter::splitterName()
            << std::endl;
  T.tic();
  FMCA::SphereClusterTree hct(P, 10);
  T.toc("H2 cluster tree:");
  const MatrixEvaluatorUS mat_eval(mom, mom, function);
  for (FMCA::Scalar eta = 0.8; eta >= 0.1; eta *= 0.5) {
    std::cout << "eta:                          " << eta << std::endl;
    T.tic();
    HMatrix hmat;
    hmat.computeHMatrix(hct, hct, mat_eval, eta, 1e-8);
    // hmat.computePattern(ct, ct, eta);
    T.toc("elapsed time:                ");
    hmat.statistics();

    {
      FMCA::Matrix X(NPTS, 10), Y1(NPTS, 10), Y2(NPTS, 10);
      X.setZero();
      X.setZero();
      for (auto i = 0; i < 10; ++i) {
        FMCA::Index index = rand() % P.cols();
        FMCA::Vector col = function.eval(P, P.col(hct.indices()[index]));
        Y1.col(i) = hct.toClusterOrder(col);
        X(index, i) = 1;
      }
      std::cout << "set test data" << std::endl;
      T.tic();
      Y2 = hmat * X;  // hmat.action(mat_eval, X);
      FMCA::Scalar err = (Y1 - Y2).norm() / Y1.norm();
      std::cout << "compression error:            " << err << std::endl;
    }
    T.toc("elapsed time:                ");
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}
