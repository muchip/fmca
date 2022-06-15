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
  for (unsigned int npts : {1e4}) {
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
    FMCA::samplet_matrix_multiplier<H2SampletTree> multip;
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
    FMCA::SparseMatrix<double> Sfast(P.cols(), P.cols());
    I.setIdentity();
    I *= 2;
    T.tic();
    S.setFromTriplets(trips.begin(), trips.end());
    T.toc("Eigen::Sparse ");
    T.tic();
    Sfast.setFromTriplets(trips.begin(), trips.end());
    T.toc("FMCA::Sparse ");
    S = 0.5 * (S + Eigen::SparseMatrix<double>(S.transpose()));
    T.tic();
    X = S.transpose() * S;
    T.toc("Eigen::Sparse mult");
    T.tic();
    FMCA::SparseMatrix<double> SS = Sfast * Sfast;
    T.toc("FMCA::Sparse mult");
    auto t2 = SS.toTriplets();
    AX.setFromTriplets(t2.begin(), t2.end());
    std::cout << "err: " << (X - AX).norm() / X.norm() << std::endl;
    X.setIdentity();
    X.diagonal().array() = 0.8 / (S.diagonal().array() + 1e-6) /
                           S.diagonal().cwiseAbs().maxCoeff();
    auto lvls = FMCA::internal::sampletLevelMapper(hst);
    Eigen::MatrixXd S0;
    Eigen::MatrixXd X0;
    Eigen::MatrixXd AX0;
    Eigen::MatrixXd X0old;
    Eigen::MatrixXd I0;

    for (auto i = 0; i < 5; ++i) {
      FMCA::IndexType level_index = 0;

      while (lvls[level_index] <= i) ++level_index;
      --level_index;

      S0 = S.block(0, 0, level_index, level_index);
      X0 = X.block(0, 0, level_index, level_index);
      I0 = I.block(0, 0, level_index, level_index);
      // if (i > 0)
      //   X0.block(0, 0, X0old.rows(), X0old.cols()) = X0old;
      double err = (X0 * S0 - 0.5 * I0).norm() / (0.5 * I0).norm();
      unsigned int iter = 0;
      while (err > 1e-6) {
        AX0 = S0 * X0;
        AX0 = I0 - AX0;
        X0 = (X0 * AX0);
        X0 = 0.5 * (X0 + X0.transpose());
        err = (X0 * S0 - 0.5 * I0).norm() / (0.5 * I0).norm();
        ++iter;
      }
      std::cout << X0.rows() << " " << X0.cols() << " "
                << X0.nonZeros() / X0.cols() << std::endl;
      std::cout << "iter: " << iter << "err " << err << std::endl;
      X0old = X0;
    }
#if 0
    T.tic();
    for (auto i = 0; i < 30; ++i) {
      AX0 = multip.multiply(hst, S0, X0, eta, 1e-4);
      AX0 = I0 - AX0;
      X0 = multip.multiply(hst, X0, AX, eta, 1e-4);
      X0 = 0.5 * (X0 + Eigen::SparseMatrix<double>(X0.transpose()));
      std::cout << (X0.transpose() * S0 - 0.5 * I0).norm() / sqrt(P.cols())
                << std::endl;
    }
#endif
    std::cout << "------------------------------------------------------\n";
  }
  return 0;
}
