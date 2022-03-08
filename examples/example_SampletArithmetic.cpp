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
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/Samplets/samplet_matrix_multiplier.h>
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>
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
  const double threshold = 1e-5;
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
    Eigen::SparseMatrix<double> S(P.cols(), P.cols());
    Eigen::SparseMatrix<double> X(P.cols(), P.cols());
    Eigen::SparseMatrix<double> AX(P.cols(), P.cols());
    Eigen::SparseMatrix<double> I(P.cols(), P.cols());
    I.setIdentity();
    I *= 2;
    S.setFromTriplets(trips.begin(), trips.end());
    S = 0.5 * (S + Eigen::SparseMatrix<double>(S.transpose()));

    X.setIdentity();
    X *= 1e-6;
    // X.diagonal().array() = 1. / (S.diagonal().array() + 1e-6);
    FMCA::IO::print2m("samp_mult.m", "S", S, "w");
    FMCA::IO::print2m("samp_mult.m", "X0", X, "a");

    FMCA::samplet_matrix_multiplier<H2SampletTree> multip;
    T.tic();
    for (auto i = 0; i < 30; ++i) {
      AX = multip.multiply(hst, S, X, eta, 0);
      AX = I - AX;
      X = multip.multiply(hst, X, AX, eta, 0);
      X = 0.5 * (X + Eigen::SparseMatrix<double>(X.transpose()));
    }
    FMCA::IO::print2m("samp_mult.m", "XAX", X, "a");

    T.toc("matrix multiplier: ");
    T.tic();
    std::cout << "------------------------------------------------------\n";
  }
  return 0;
}
