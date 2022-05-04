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

#include "pardiso_interface.h"
////////////////////////////////////////////////////////////////////////////////
struct expKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    const double r = (x - y).norm();
    const double ell = 1;
    return exp(-r / ell);
    //return (1 + sqrt(3) * r / ell) * exp(-sqrt(3) * r / ell);
  }
};

struct rationalQuadraticKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    const double r = (x - y).norm();
    constexpr double alpha = 0.5;
    constexpr double ell = 2.;
    constexpr double c = 1. / (2. * alpha * ell * ell);
    return std::pow(1 + c * r * r, -alpha);
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
  std::vector<Eigen::Triplet<double>> trips = comp.pattern_triplets();
  std::vector<Eigen::Triplet<double>> inv_trips =
      FMCA::symPattern(hst, inv_eta);
  FMCA::SparseMatrix<double>::sortTripletsInPlace(trips);
  FMCA::SparseMatrix<double>::sortTripletsInPlace(inv_trips);
  std::cout << "compression error: "
            << FMCA::errorEstimatorSymmetricCompressor(trips, function, hst, P)
            << std::endl
            << std::flush;
  double lambda_max = 0;
  {
    Eigen::MatrixXd x = Eigen::VectorXd::Random(npts);
    x /= x.norm();
    for (auto i = 0; i < 20; ++i) {
      x = FMCA::SparseMatrix<double>::symTripletsTimesVector(trips, x);
      lambda_max = x.norm();
      x /= lambda_max;
    }
    std::cout << "lambda_max (est by 20its of power it): " << lambda_max
              << std::endl;
  }
  for (auto &&it : trips) {
    if (it.row() == it.col())
      it = Eigen::Triplet<double>(it.row(), it.col(), it.value() + 1e-4 * lambda_max);
  }
  // merge trips into inv_trips
  unsigned int j = 0;
  for (auto i = 0; i < trips.size(); ++i) {
    while (trips[i].row() != inv_trips[j].row() ||
           trips[i].col() != inv_trips[j].col())
      ++j;
    inv_trips[j] = trips[i];
  }
  std::cout << "entries A:          "
            << 100 * double(trips.size()) / npts / npts << "\%" << std::endl;
  std::cout << std::string(75, '=') << std::endl;
  {
    int i = 0;
    int j = 0;
    int n = npts;
    int n_triplets = inv_trips.size();
    int *ia = nullptr;
    int *ja = nullptr;
    double *a = nullptr;
    ia = (int *)malloc((n + 1) * sizeof(int));
    ja = (int *)malloc(n_triplets * sizeof(int));
    a = (double *)malloc(n_triplets * sizeof(double));
    memset(ia, 0, (n + 1) * sizeof(int));
    memset(ja, 0, n_triplets * sizeof(int));
    memset(a, 0, n_triplets * sizeof(double));
    // write rows
    ia[trips[0].row()] = 0;
    for (i = trips[0].row() + 1; i <= n; ++i) {
      while (j < n_triplets && i - 1 == inv_trips[j].row())
        ++j;
      ia[i] = j;
    }
    // write the rest
    for (i = 0; i < n_triplets; ++i) {
      ja[i] = inv_trips[i].col();
      a[i] = inv_trips[i].value();
    }
    std::cout << "\n\nentering pardiso block" << std::flush;
    T.tic();
    pardiso_interface(ia, ja, a, n);
    std::cout << std::string(75, '=') << std::endl;
    T.toc("Wall time pardiso: ");
    inv_trips.clear();
    for (i = 0; i < n; ++i)
      for (j = ia[i]; j < ia[i + 1]; ++j)
        if (abs(a[j]) > 1e-8)
          inv_trips.push_back(Eigen::Triplet<double>(i, ja[j], a[j]));
    free(ia);
    free(ja);
    free(a);
  }
  std::cout << "inverse entries:    " << 100. * inv_trips.size() / npts / npts
            << "\%" << std::endl;
  {
    Eigen::MatrixXd rand = Eigen::MatrixXd::Random(npts, 100);
    Eigen::VectorXd nrms = rand.colwise().norm();
    for (auto i = 0; i < rand.cols(); ++i)
      rand.col(i) /= nrms(i);
    auto Srand =
        FMCA::SparseMatrix<double>::symTripletsTimesVector(trips, rand);
    auto Rrand =
        FMCA::SparseMatrix<double>::symTripletsTimesVector(inv_trips, Srand);
    std::cout << "inverse error:      " << (rand - Rrand).norm() / rand.norm()
              << std::endl;
  }
  {
    Eigen::VectorXd x = Eigen::VectorXd::Random(npts);
    Eigen::VectorXd xold;
    Eigen::VectorXd y;
    x /= x.norm();
    for (auto i = 0; i < 50; ++i) {
      xold = x;
      y = FMCA::SparseMatrix<double>::symTripletsTimesVector(trips, xold);
      x = FMCA::SparseMatrix<double>::symTripletsTimesVector(inv_trips, y);
      x -= xold;
      lambda_max = x.norm();
      x /= lambda_max;
    }
    std::cout << "op. norm err:       " << lambda_max << std::endl;
  }
  return 0;
}
