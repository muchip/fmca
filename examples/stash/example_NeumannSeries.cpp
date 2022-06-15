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
#include <algorithm>
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>

template <typename T>
struct customLess {
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
    return exp(-0.5 * (x - y).norm());
  }
};

using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, expKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main(int argc, char *argv[]) {
  const unsigned int dim = 2;
  const unsigned int dtilde = 2;
  const auto function = expKernel();
  const double eta = 0.2;
  const unsigned int mp_deg = 3;
  const double threshold = 0;
  const unsigned int npts = 5e3;
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
  T.toc("compressor: ");
  const auto &trips = comp.pattern_triplets();
  std::vector<unsigned int> lvls = FMCA::internal::sampletLevelMapper(hst);
  std::vector<unsigned int> idcs(lvls.size());
  std::iota(idcs.begin(), idcs.end(), 0);
  std::stable_sort(idcs.begin(), idcs.end(),
                   customLess<std::vector<unsigned int>>(lvls));
  //////////////////////////////////////////////////////////////////////////////
  // we sort the entries in the matrix according to the samplet diameter
  // next, we can access the levels by means of the lvlsteps array
  FMCA::SparseMatrix<double> S(P.cols(), P.cols());
  FMCA::SparseMatrix<double> X(P.cols(), P.cols());
  FMCA::SparseMatrix<double> Sp(P.cols(), P.cols());
  FMCA::SparseMatrix<double> Sp_formatted(P.cols(), P.cols());
  FMCA::SparseMatrix<double> Perm(P.cols(), P.cols());
  S.setFromTriplets(trips.begin(), trips.end());
  S.symmetrize();
  Eigen::VectorXd init(S.rows());
  for (auto i = 0; i < init.size(); ++i) init(i) = 1. / sqrt(S(i, i) + 1e-6);
  X.setDiagonal(init);
  //S = (X * S) * X;
  Sp = S;
  Sp_formatted = S;
  Eigen::MatrixXd randFilter = Eigen::MatrixXd::Random(S.rows(), 20);
  Eigen::MatrixXd K;
  mat_eval.compute_dense_block(hst, hst, &K);
  Eigen::MatrixXd KS = K;
  hst.sampletTransformMatrix(KS);
  KS = 0.5 * (KS + KS.transpose());
  //KS = X.full() * KS * X.full();
  Eigen::MatrixXd Kp = KS;
  std::cout << " err: "
            << ((Sp * randFilter) - (Kp * randFilter)).norm() /
                   (Kp * randFilter).norm()
            << std::endl
            << std::flush;

  for (auto i = 0; i < 40; ++i) {
    T.tic();
    Sp = S * Sp;
    Sp.symmetrize();
    const double tunformatted = T.toc();
    T.tic();
    Sp_formatted =
        FMCA::SparseMatrix<double>::formatted_ABT(S, S, Sp_formatted);
    const double tformatted = T.toc();
    Kp = KS * Kp;
    std::cout << "time Sp: " << tunformatted
              << " \ttime Sp_formatted: " << tformatted << " \terr: "
              << ((Sp_formatted * randFilter) - (Kp * randFilter)).norm() /
                     (Kp * randFilter).norm()
              << std::endl
              << std::flush;
  }
  std::cout << "------------------------------------------------------\n";
  return 0;
}
