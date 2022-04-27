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
#include <Eigen/Dense>
#include <Eigen/Sparse>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/HaltonSet.h>
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/Tictoc.h>

#include "pardiso_interface.h"
////////////////////////////////////////////////////////////////////////////////
struct expKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-10 * (x - y).norm());
  }
};
////////////////////////////////////////////////////////////////////////////////
using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, expKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const unsigned int dtilde = 4;
  const auto function = expKernel();
  const double eta = 0.8;
  const unsigned int mp_deg = 6;
  const double threshold = 0;
  const unsigned int dim = 2;
  const unsigned int npts = 160000;
  FMCA::HaltonSet<100> hs(dim);
  FMCA::Tictoc T;
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(dim, npts);
  for (auto i = 0; i < P.cols(); ++i) {
    Eigen::VectorXd bla = hs.EigenHaltonVector();
    P.col(i) = bla;
    hs.next();
  }
  std::cout << P.leftCols(6) << std::endl;
  std::cout << std::string(75, '=') << std::endl;
  std::cout << "npts:   " << npts << std::endl
            << "dim:    " << dim << std::endl
            << "dtilde: " << dtilde << std::endl
            << "mp_deg: " << mp_deg << std::endl
            << "eta:    " << eta << std::endl
            << std::flush;
  const Moments mom(P, mp_deg);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("tree setup:        ");
  FMCA::unsymmetric_compressor_impl<H2SampletTree> comp;
  T.tic();
  comp.compress(hst, mat_eval, eta, threshold);
  T.toc("compressor:        ");
  T.tic();
  const auto &trips = comp.pattern_triplets();
  FMCA::SparseMatrix<double> S(npts, npts);
  FMCA::SparseMatrix<double> S2(npts, npts);
  S.setFromTriplets(trips.begin(), trips.end());
  std::cout << "entries A (\%): " << 100 * double(S.nnz()) / npts / npts
            << std::endl;
  S.symmetrize();
  T.toc("sparse matrix:     ");
  T.tic();
  for (auto i = 0; i < 1; ++i) {
    S2 = FMCA::SparseMatrix<double>::formatted_ABT(S, S, S);
  }
  T.toc("time 10 mat mult:  ");
  Eigen::MatrixXd rand = Eigen::MatrixXd::Random(npts, 100);
  auto Srand = S * (S * rand);
  auto Rrand = S2 * rand;
  std::cout << "error: " << (Srand - Rrand).norm() / Srand.norm() << std::endl;
}
