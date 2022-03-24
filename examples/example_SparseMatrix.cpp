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
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/Tictoc.h>

struct expKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-(x - y).norm());
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
  const double threshold = 1e-8;
  FMCA::Tictoc T;
  for (unsigned int npts : {1e3}) {
    // for (unsigned int npts : {5e6}) {
    std::cout << "N:" << npts << " dim:" << dim << " eta:" << eta
              << " mpd:" << mp_deg << " dt:" << dtilde
              << " thres: " << threshold << std::endl;
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
    Eigen::MatrixXd S(P.cols(), P.cols());
    FMCA::SparseMatrix<double> S2(P.cols(), P.cols());
    FMCA::SparseMatrix<double> I(P.cols(), P.cols());
    FMCA::SparseMatrix<double> I2(P.cols(), P.cols());
    FMCA::SparseMatrix<double> X(P.cols(), P.cols());
    S.setZero();
    for (auto it : trips)
      S(it.row(), it.col()) = it.value();
    T.tic();
    S2.setFromTriplets(trips.begin(), trips.end());
    T.toc("FMCA::Sparse ");
    for (auto i = 0; i < I.rows(); ++i) {
      I(i, i) = 1;
      I2(i, i) = 2;
      X(i, i) = 0.25 / (S2(i,i) + 1);
    }
    for (auto i = 0; i < 40; ++i) {
      FMCA::SparseMatrix<double> IXS = I2 - (X * S2);
      X = X * IXS;
      X.symmetrize();
      std::cout << ((S2 * X) - I).full().norm() << std::endl;
    }
  }
  return 0;
}
