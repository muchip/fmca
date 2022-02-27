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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>

#include "../FMCA/src/util/Errors.h"
#include "../FMCA/src/util/print2file.h"
#include <FMCA/src/util/tictoc.hpp>

#include <FMCA/Clustering>
#include <FMCA/H2Matrix>
#include <FMCA/MatrixEvaluators>

struct expKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-(x - y).norm());
  }
};

int main(int argc, char *argv[]) {
  const auto function = expKernel();
  tictoc T;

  using Moments =
      FMCA::NystromMoments<FMCA::TotalDegreeInterpolator<FMCA::FloatType>>;
  using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, expKernel>;
  using H2ClusterT = FMCA::H2ClusterTree<FMCA::ClusterT>;
  using H2Matrix = FMCA::H2Matrix<H2ClusterT>;
  std::cout << std::string(60, '-') << std::endl;
  for (auto i = 2; i < 7; ++i) {
    const unsigned int npts = std::pow(10, i);
    const Eigen::MatrixXd P = Eigen::MatrixXd::Random(2, npts);
    Moments nyst_mom(P, 3);
    MatrixEvaluator mat_eval(nyst_mom, function);
    H2ClusterT H2CT(nyst_mom, 1, P);
    T.tic();
    H2Matrix H2mat(H2CT, mat_eval, 0.8);
    const double tset = T.toc("matrix setup: ");
    {
      Eigen::VectorXd x(npts), y1(npts), y2(npts);
      double err = 0;
      double nrm = 0;
      for (auto i = 0; i < 10; ++i) {
        unsigned int index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        y1 = FMCA::matrixColumnGetter(P, H2CT.indices(), function, index);
        y2 = H2mat * x;
        err += (y1 - y2).squaredNorm();
        nrm += y1.squaredNorm();
      }
      err = sqrt(err / nrm);
      std::cout << "compression error: " << err << std::endl;
      // (m, n, fblocks, lrblocks, nz(A), mem)
      const std::vector<double> stats = H2mat.get_statistics();
      std::cout << std::string(60, '-') << std::endl;
    }
  }

  return 0;
}
