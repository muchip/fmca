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
#include <FMCA/src/Samplets/samplet_matrix_multiplier.h>
#include <FMCA/src/util/Errors.h>
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
  const double threshold = 0;
  const unsigned int npts = 1e3;
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
  FMCA::samplet_matrix_multiplier<H2SampletTree> multip;
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("tree setup: ");
  FMCA::unsymmetric_compressor_impl<H2SampletTree> comp;
  T.tic();
  comp.compress(hst, mat_eval, eta, threshold);
  const double tcomp = T.toc("compressor: ");
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
  FMCA::SparseMatrix<double> Xold(P.cols(), P.cols());
  FMCA::SparseMatrix<double> Perm(P.cols(), P.cols());
  // FMCA::SparseMatrix<double> Pat(Eigen::MatrixXd::Ones(P.cols(), P.cols()));
  // Eigen::MatrixXd K;
  // mat_eval.compute_dense_block(hst, hst, &K);
  // Eigen::MatrixXd KS = K;
  // hst.sampletTransformMatrix(KS);
  S.setFromTriplets(trips.begin(), trips.end());
  Perm.setPermutation(idcs);
  // S = FMCA::SparseMatrix<double>(KS);
  S.symmetrize();
  S = Perm * S;
  S = S * Perm.transpose();
  Perm = Perm.transpose();
  Eigen::VectorXd lvl(idcs.size());
  std::vector<unsigned int> lvlsteps;
  unsigned int jump = 0;
  for (auto i = 0; i < lvl.size(); ++i) {
    lvl(i) = lvls[idcs[i]];
    if (jump < lvl(i)) {
      lvlsteps.push_back(i);
      jump = lvl(i);
    }
  }
  Eigen::VectorXd init(S.rows());
  for (auto i = 0; i < init.size(); ++i)
    init(i) = 1. / sqrt(S(i, i) + 1e-6);
  X.setDiagonal(init);
  S = (X * S) * X;
  FMCA::IO::print2m("matrices.m", "S", S.full(), "w");

  for (auto lvl = 0; lvl < lvlsteps.size(); ++lvl) {
    std::cout << "block size: " << lvlsteps[lvl] << std::endl;
    FMCA::SparseMatrix<double> Slvl = S;
    Slvl.resize(lvlsteps[lvl], lvlsteps[lvl]);
    FMCA::SparseMatrix<double> I2(Slvl.rows(), Slvl.cols());

    if (lvl == 0) {
      X.resize(Slvl.rows(), Slvl.cols());
      X.setZero();
      for (auto i = 0; i < X.rows(); ++i)
        X(i, i) = 0.25;
    } else {
      X = Xold;
      X.resize(Slvl.rows(), Slvl.rows());
      for (auto i = lvlsteps[lvl - 1]; i < lvlsteps[lvl]; ++i)
        X(i, i) = 0.25;
    }
    I2.setDiagonal(2 * Eigen::VectorXd::Ones(Slvl.rows()));
    Eigen::MatrixXd randFilter = Eigen::MatrixXd::Random(Slvl.rows(), 20);
    for (auto i = 0; i < 10; ++i) {
      T.tic();
      X = (I2 * X) - (X * (Slvl * X));
      // X = (I2 * X) - FMCA::SparseMatrix<double>::formatted_BABT(Slvl, Slvl,
      // X);
      T.toc("Schulz step: ");
      std::cout << "a priori anz: " << X.nnz() / npts;
      X.compress(1e-6);
      X.symmetrize();
      std::cout << "  a post anz: " << X.nnz() / npts;
      std::cout << "  err: "
                << ((X * (Slvl * randFilter)) - randFilter).norm() /
                       randFilter.norm()
                << std::endl
                << std::flush;
    }
    std::cout << "a priori anz: " << X.nnz() / npts;
    X.compress(1e-4);
    X.symmetrize();
    std::cout << "  a post anz: " << X.nnz() / npts << std::endl;
    std::cout << "  err: "
              << ((X * (Slvl * randFilter)) - randFilter).norm() /
                     randFilter.norm()
              << std::endl
              << std::flush;
    Xold = X;
    FMCA::IO::print2m("matrices.m", "X" + std::to_string(lvl), X.full(), "a");
  }
  std::cout << "------------------------------------------------------\n";
  return 0;
}
