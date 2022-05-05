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
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/HaltonSet.h>
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/Tictoc.h>

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
  typedef std::vector<Eigen::Triplet<double>> TripletVector;
  const unsigned int dtilde = 4;
  const auto function = expKernel();
  const double eta = atof(argv[2]);
  const double inv_eta = atof(argv[3]);
  const unsigned int mp_deg = 6;
  const double threshold = 1e-6;
  const unsigned int dim = 2;
  const unsigned int npts = atoi(argv[1]);
  FMCA::HaltonSet<100> hs(dim);
  FMCA::Tictoc T;
  Eigen::MatrixXd P = 0.5 * (Eigen::MatrixXd::Random(dim, npts).array() + 1);
#if 0
  for (auto i = 0; i < P.cols(); ++i) {
    P.col(i) = hs.EigenHaltonVector();
    hs.next();
  }
#endif
  std::cout << std::string(75, '=') << std::endl;
  std::cout << "npts: " << npts << " | dim: " << dim << " | dtilde: " << dtilde
            << " | mp_deg: " << mp_deg << " | eta: " << eta
            << " | inv_eta: " << inv_eta << std::endl
            << std::flush;
  const Moments mom(P, mp_deg);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("tree setup:        ");
  std::cout << "fill_dist: " << FMCA::fillDistance(hst, P) << std::endl;
  std::cout << "sep_rad: " << FMCA::separationRadius(hst, P) << std::endl;
  std::cout << "bb: " << std::endl << hst.bb() << std::endl;
  FMCA::symmetric_compressor_impl<H2SampletTree> comp;
  T.tic();
  comp.compress(hst, mat_eval, eta, threshold);
  const double tcomp = T.toc("compressor:        ");
  T.tic();
  TripletVector sym_trips = comp.pattern_triplets();
  TripletVector inv_trips = FMCA::symPattern(hst, inv_eta);
  FMCA::SparseMatrix<double>::sortTripletsInPlace(sym_trips);
  FMCA::SparseMatrix<double>::sortTripletsInPlace(inv_trips);
  FMCA::SparseMatrix<double> pattern(npts, npts);
  FMCA::SparseMatrix<double> S(npts, npts);
  FMCA::SparseMatrix<double> I2(npts, npts);
  pattern.setFromTriplets(inv_trips.begin(), inv_trips.end());
  S.setFromTriplets(sym_trips.begin(), sym_trips.end());
  S.mirrorUpper();
  pattern.mirrorUpper();
  std::cout << "compression error:  "
            << FMCA::errorEstimatorSymmetricCompressor(sym_trips, function, hst,
                                                       P)
            << std::endl
            << std::flush;
  double lambda_max = 0;
  {
    Eigen::MatrixXd x = Eigen::VectorXd::Random(npts);
    x /= x.norm();
    for (auto i = 0; i < 20; ++i) {
      x = FMCA::SparseMatrix<double>::symTripletsTimesVector(sym_trips, x);
      lambda_max = x.norm();
      x /= lambda_max;
    }
    std::cout << "lambda_max:         " << lambda_max << std::endl;
  }
  std::cout << "entries A:          "
            << 100 * double(sym_trips.size()) / npts / npts << "\%"
            << std::endl;
  double trace = 0;
  Eigen::VectorXd diag(npts);
  for (auto i = 0; i < S.rows(); ++i) {
    S(i, i) += 1e-4 * lambda_max;
    diag(i) = S(i, i);
  }
  std::vector<FMCA::IndexType> idcs(npts);
  {
    std::iota(idcs.begin(), idcs.end(), 0);
    struct customGtr {
      customGtr(const Eigen::VectorXd &vec) : vec_(vec) {}
      bool operator()(FMCA::IndexType a, FMCA::IndexType b) {
        return vec_(a) > vec_(b);
      }
      const Eigen::VectorXd &vec_;
    };
    std::sort(idcs.begin(), idcs.end(), customGtr(diag));
  }
  //////////////////////////////////////////////////////////////////////////////
  I2.setPermutation(idcs);
  S = I2 * S;
  pattern = I2 * pattern;
  S = S * I2.transpose();
  pattern = pattern * I2;
  FMCA::IndexType ind = 0;
  for (ind = 0; S(ind, ind) >= S(0, 0) * 5e-4; ++ind)
    ;
  //////////////////////////////////////////////////////////////////////////////
  std::cout << "block size low pass: " << ind << std::endl << std::flush;
  std::cout << std::string(75, '=') << std::endl;
  FMCA::SparseMatrix<double> Ssmall;
  FMCA::SparseMatrix<double> Psmall;
  FMCA::SparseMatrix<double> X(ind, ind);
  FMCA::SparseMatrix<double> Xl(ind, ind);
  FMCA::SparseMatrix<double> Xr(ind, ind);
  FMCA::SparseMatrix<double> ImXS(ind, ind);
  Ssmall = S;
  Psmall = pattern;
  Ssmall.resize(ind, ind);
  Psmall.resize(ind, ind);
  ImXS = Psmall;
  Xl = Psmall;
  Xr = Psmall;
  double err = 10;
  double err_old = 10;
  X.setIdentity().scale(0.01 / lambda_max);
  Eigen::MatrixXd randFilter = Eigen::MatrixXd::Random(ind, 10);
  Eigen::VectorXd rscale = randFilter.colwise().norm();
  for (auto i = 0; i < randFilter.cols(); ++i)
    randFilter.col(i) /= rscale(i);
  I2.setIdentity().scale(2);
  std::cout << "matrices initialized\n" << std::flush;
  //////////////////////////////////////////////////////////////////////////////
  T.tic();
  for (auto inner_iter = 0; inner_iter < 200; ++inner_iter) {
    FMCA::SparseMatrix<double>::formatted_ABT(ImXS, X, Ssmall);
    ImXS.scale(-1);
#pragma omp parallel for
    for (auto i = 0; i < ImXS.rows(); ++i)
      ImXS(i, i) += 2;
    FMCA::SparseMatrix<double>::formatted_ABT(Xl, ImXS, X);
    FMCA::SparseMatrix<double>::formatted_ABT(Xr, X, ImXS);
    X = std::move((Xl + Xr).scale(0.5));
    err_old = err;
    Eigen::MatrixXd bla1 = Ssmall * randFilter;
    Eigen::MatrixXd bla2 = X * bla1;
    err = (bla2 - randFilter).norm() / randFilter.norm();
  std::cout << err << std::endl;
  if (err > err_old) {
      break;
    }
  }
  T.toc("time initial guess: ");
  std::cout << "initial error: " << err << std::endl;
  std::cout << std::string(75, '=') << "\n";
  //////////////////////////////////////////////////////////////////////////////
  X.resize(npts, npts);
  for (auto i = ind; i < X.rows(); ++i)
    X(i, i) = 1;
  ImXS = pattern;
  Xl = pattern;
  Xr = pattern;
  randFilter = Eigen::MatrixXd::Random(npts, 10);
  rscale = randFilter.colwise().norm();
  for (auto i = 0; i < randFilter.cols(); ++i)
    randFilter.col(i) /= rscale(i);
  err = err_old = 10;
  //////////////////////////////////////////////////////////////////////////////
  T.tic();
  for (auto inner_iter = 0; inner_iter < 200; ++inner_iter) {
    FMCA::SparseMatrix<double>::formatted_ABT(ImXS, X, S);
    ImXS.scale(-1);
#pragma omp parallel for
    for (auto i = 0; i < ImXS.rows(); ++i)
      ImXS(i, i) += 2;
    FMCA::SparseMatrix<double>::formatted_ABT(Xl, ImXS, X);
    FMCA::SparseMatrix<double>::formatted_ABT(Xr, X, ImXS);
    X = std::move((Xl + Xr).scale(0.5));
    err_old = err;
    Eigen::MatrixXd bla1 = S * randFilter;
    Eigen::MatrixXd bla2 = X * bla1;
    err = (bla2 - randFilter).norm() / randFilter.norm();
    std::cout << "anz: " << X.nnz() / S.rows() << " err: " << err << std::endl
              << std::flush;
    if (err > err_old) {
      break;
    }
  }
  T.toc("time Schulz iteration: ");
  {
    Eigen::MatrixXd x = Eigen::VectorXd::Random(npts, 1);
    Eigen::MatrixXd xold;
    Eigen::MatrixXd y;
    x /= x.norm();
    for (auto i = 0; i < 50; ++i) {
      xold = x;
      y = S * xold;
      x = X * y;
      x -= xold;
      lambda_max = x.norm();
      x /= lambda_max;
    }
    std::cout << "op. norm err: " << lambda_max << std::endl;
  }

  return 0;
}
