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
#define FMCA_CLUSTERSET_
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
/////////////////////////
#include <Eigen/Dense>
#include <FMCA/Samplets>

#include "../FMCA/src/util/Errors.h"
#include "../FMCA/src/util/tictoc.hpp"
#include "generateSwissCheese.h"
#include "generateSwissCheeseExp.h"

struct exponentialKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-(x - y).norm());
  }
};

struct rationalQuadraticKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    const double r = (x - y).norm();
    constexpr double alpha = 0.5;
    constexpr double ell = 1.;
    constexpr double c = 1. / (2. * alpha * ell * ell);
    return std::pow(1 + c * r * r, -alpha);
  }
};

using theKernel = exponentialKernel;
using theH2Matrix = FMCA::H2Matrix<FMCA::H2ClusterTree>;

const double parameters[4][3] = {
    {2, 1, 1e-2}, {3, 2, 1e-3}, {4, 3, 1e-4}, {6, 4, 1e-5}};

int main(int argc, char *argv[]) {
  const unsigned int dim = atoi(argv[1]);
  const unsigned int dtilde = atoi(argv[2]);
  const auto function = theKernel();
  const double eta = 0.8;
  const unsigned int mp_deg = parameters[dtilde - 1][0];
  const double threshold = parameters[dtilde - 1][2];
  tictoc T;
  std::fstream file;
  file.open(
      "H2_output_SC_" + std::to_string(dim) + "_" + std::to_string(mp_deg) + ".txt",
      std::ios::out);
  file << "m           n     fblocks    lrblocks       nz(A)";
  file << "         mem         err\n" << std::flush;
  for (unsigned int npts : {1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6}) {
    std::cout << "N:" << npts << " dim:" << dim << " eta:" << eta
              << " mpd:" << mp_deg << " dt:" << dtilde
              << " thres: " << threshold << std::endl
              << std::flush;
    T.tic();
    const Eigen::MatrixXd P = generateSwissCheese(dim, npts);
    T.toc("geometry generation: ");
    T.tic();
    const FMCA::H2ClusterTree H2CT(P, 1, mp_deg);
    T.toc("Cluster tree generation: ");
    const FMCA::NystromMatrixEvaluator<FMCA::H2ClusterTree, theKernel> nm_eval(
        P, function);
    FMCA::symmetric_compressor_impl<FMCA::H2SampletTree> symComp;
    T.tic();
    FMCA::H2Matrix<FMCA::H2ClusterTree> H2mat(H2CT, nm_eval, eta);
    const double tcomp = T.toc("H2matrix setup: ");
    std::cout << std::flush;
    // (m, n, fblocks, lrblocks, nz(A), mem)
    const std::vector<double> stats = H2mat.get_statistics();
    for (const auto &it : stats)
      file << std::setw(10) << std::setprecision(6) << it << "\t";
    file << std::flush;
    {
      Eigen::VectorXd x(npts), y1(npts), y2(npts);
      double err = 0;
      double nrm = 0;
      T.tic();
      for (auto i = 0; i < 100; ++i) {
        unsigned int index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        y1 = FMCA::matrixColumnGetter(P, H2CT.indices(), function, index);
        y2 = H2mat * x;
        err += (y1 - y2).squaredNorm();
        nrm += y1.squaredNorm();
      }
      const double thet = T.toc("time matrix vector: ");
      std::cout << thet / 100 << std::endl;
      err = sqrt(err / nrm);
      std::cout << "compression error: " << err << std::endl;

      file << std::setw(10) << std::setprecision(6) << err << "\n"
           << std::flush;
    }

    std::cout << std::string(60, '-') << std::endl;
  }
  file.close();
  return 0;
}
