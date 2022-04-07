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
#define FMCA_CLUSTERSET_
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>

template <typename T> struct customLess {
  customLess(const T &array) : array_(array) {}
  bool operator()(typename T::size_type a, typename T::size_type b) const {
    return array_[a] < array_[b];
  }
  const T &array_;
};

struct expKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-1 * (x - y).norm());
  }
};

using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, expKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main(int argc, char *argv[]) {
  const unsigned int dim = atoi(argv[1]);
  const unsigned int dtilde = 3;
  const auto function = expKernel();
  const double eta = 0.8;
  const unsigned int mp_deg = 4;
  const double threshold = atof(argv[2]);
  const unsigned int npts = 1e4;
  FMCA::Tictoc T;
  std::cout << "N:" << npts << " dim:" << dim << " eta:" << eta
            << " mpd:" << mp_deg << " dt:" << dtilde << " thres: " << threshold
            << std::endl;
  T.tic();
  const Eigen::MatrixXd P = Eigen::MatrixXd::Random(dim, npts);
  T.toc("geometry generation: ");
  const Moments mom(P, mp_deg);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("tree setup: ");
  FMCA::unsymmetric_compressor_impl<H2SampletTree> comp;
  T.tic();
  comp.compress(hst, mat_eval, eta, threshold);
  const double tcomp = T.toc("compressor: ");
  const auto &trips = comp.pattern_triplets();
  //////////////////////////////////////////////////////////////////////////////
  // we sort the entries in the matrix according to the samplet diameter
  // next, we can access the levels by means of the lvlsteps array
  FMCA::SparseMatrix<double> S(P.cols(), P.cols());
  FMCA::SparseMatrix<double> X(P.cols(), P.cols());
  FMCA::SparseMatrix<double> Xold(P.cols(), P.cols());
  FMCA::SparseMatrix<double> I2(P.cols(), P.cols());
  Eigen::MatrixXd randFilter = Eigen::MatrixXd::Random(S.rows(), 20);
  S.setFromTriplets(trips.begin(), trips.end());
  S.symmetrize();
  double lambda_max = 0;
  {
    Eigen::MatrixXd x = Eigen::VectorXd::Random(S.cols());
    x /= x.norm();
    for (auto i = 0; i < 20; ++i) {
      x = S * x;
      lambda_max = x.norm();
      x /= lambda_max;
    }
    std::cout << "lambda_max (est by 20its of power it): " << lambda_max
              << std::endl;
  }
  double trace = 0;
  for (auto i = 0; i < S.rows(); ++i)
    trace += S(i, i);
  std::cout << "trace: " << trace << " anz: " << S.nnz() / S.cols()
            << std::endl;
  std::cout << "norm: " << S.norm() << std::endl;
  double alpha = 1. / lambda_max / lambda_max;
  std::cout << "chosen alpha for initial guess: " << alpha << std::endl;
  double err = 10;
  double err_old = 10;
  X = S;
  X.scale(alpha);
  std::cout << "initial guess: "
            << ((X * (S * randFilter)) - randFilter).norm() / randFilter.norm()
            << std::endl;
  I2.setIdentity().scale(2);
  for (auto inner_iter = 0; inner_iter < 100; ++inner_iter) {
    // ImXS = I2 - FMCA::SparseMatrix<double>::formatted_ABT(S, X, Seps);
    // X = FMCA::SparseMatrix<double>::formatted_ABT(S, X, ImXS);
    X = I2 * X - FMCA::SparseMatrix<double>::formatted_BABT(S, S, X);
    err_old = err;
    err = ((X * (S * randFilter)) - randFilter).norm() / randFilter.norm();
    std::cout << "err: " << err << std::endl;
    if (err > err_old) {
      X = Xold;
      break;
    }
  }

  T.toc("time inner: ");
  std::cout << std::string(60, '=') << "\n";

  return 0;
}
