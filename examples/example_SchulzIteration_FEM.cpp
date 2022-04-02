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
#include "../Points/matrixReader.h"
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
  const unsigned int dtilde = 4;
  const auto function = expKernel();
  const double eta = 0.8;
  const unsigned int mp_deg = 6;
  const double threshold = atof(argv[1]);
  FMCA::Tictoc T;
  T.tic();
  const Eigen::MatrixXd P =
      readMatrix("../mex/mfiles/P_mb_0003.txt").transpose();
  const Eigen::MatrixXd Atrips = readMatrix("../mex/mfiles/A_mb_0003.txt");
  T.toc("geometry generation: ");
  const unsigned int npts = P.cols();
  const unsigned int dim = P.rows();
  std::cout << "N:" << npts << " dim:" << dim << " eta:" << eta
            << " mpd:" << mp_deg << " dt:" << dtilde << " thres: " << threshold
            << std::endl;

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
  FMCA::SparseMatrix<double> Seps(P.cols(), P.cols());
  FMCA::SparseMatrix<double> X(P.cols(), P.cols());
  FMCA::SparseMatrix<double> Xold(P.cols(), P.cols());
  FMCA::SparseMatrix<double> I2(P.cols(), P.cols());
  Eigen::MatrixXd randFilter = Eigen::MatrixXd::Random(S.rows(), 20);
  S.setFromTriplets(trips.begin(), trips.end());
  S.symmetrize();
  double trace = 0;
  for (auto i = 0; i < S.rows(); ++i)
    trace += S(i, i);
  std::cout << "trace: " << trace << " anz: " << S.nnz() / S.cols()
            << std::endl;
  for (auto outer_iter = 0; outer_iter < 2; ++outer_iter) {
    double reg = 10. / (1 << outer_iter);
    std::cout << "regularization: " << reg << std::endl;
    I2.setIdentity().scale(reg);
    Seps = S + I2;
    I2.setIdentity().scale(2);
    if (outer_iter == 0) {
      Eigen::VectorXd inv_diagS(Seps.rows());
      for (auto i = 0; i < Seps.rows(); ++i)
        inv_diagS(i) = 0.5 / Seps(i, i);
      X.setDiagonal(inv_diagS);
      I2.setIdentity().scale(2);
    } else {
      // It holds (A+E)^-1\approx A^-1-A^-1EA^-1
      // Thus letting A = S + c_1I and E = - c_2I, we get
      // (A+(c1-c_2)I)^-1 = (A+c_1I)^-1+c_2(A+c_1I)^-2
      // X = X + FMCA::SparseMatrix<double>::formatted_BABT(S, X, X).scale(reg);
    }
    T.tic();
    for (auto inner_iter = 0; inner_iter < 10; ++inner_iter) {
      // X = (I2 * X) - (X * (Seps * X));
      Xold = I2 - FMCA::SparseMatrix<double>::formatted_ABT(S, X, Seps);
      X = FMCA::SparseMatrix<double>::formatted_ABT(S, X, Xold);
      X.symmetrize();
      std::cout << "err: "
                << ((X * (Seps * randFilter)) - randFilter).norm() /
                       randFilter.norm()
                << std::endl
                << std::flush;
    }
    T.toc("time inner: ");
    std::cout << "------------------------------------------------------\n";
  }

  return 0;
}
