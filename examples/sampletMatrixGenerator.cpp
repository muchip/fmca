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
#include <limits>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/Sparse>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/Samplets/omp_samplet_compressor.h>
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/Tictoc.h>

#include "sampletMatrixGenerator.h"

////////////////////////////////////////////////////////////////////////////////
struct expKernel {
  expKernel(const FMCA::Index n) : n_(n) {}
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    const double r = (x - y).norm();
    const double ell = 1;
    return 1. / n_ * exp(-r / ell);
    // return (1 + sqrt(3) * r / ell) * exp(-sqrt(3) * r / ell);
  }
  const FMCA::Index n_;
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
using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, expKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
////////////////////////////////////////////////////////////////////////////////
CRSmatrix
sampletMatrixGenerator(const Eigen::MatrixXd &P, const unsigned int mp_deg = 4,
                       const unsigned int dtilde = 3, const double eta = 0.8,
                       double threshold = 1e-4, const double ridgep = 1e-6) {
  typedef std::vector<Eigen::Triplet<double>> TripletVector;
  CRSmatrix retval;
  const unsigned int npts = P.cols();
  const unsigned int dim = P.rows();
  const auto function = expKernel(npts);

  threshold /= npts;
  FMCA::Tictoc T;
  std::cout << std::string(75, '=') << std::endl;
  std::cout << "N:                           " << npts << std::endl
            << "dim:                         " << dim << std::endl
            << "eta:                         " << eta << std::endl
            << "multipole degree:            " << mp_deg << std::endl
            << "vanishing moments:           " << dtilde << std::endl
            << "aposteriori threshold:       " << threshold << std::endl
            << "ridge parameter:             " << ridgep << std::endl;

  const Moments mom(P, mp_deg);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("tree setup:                 ");
  FMCA::Vector min_dist = minDistanceVector(hst, P);
  FMCA::Scalar min_min_dist = min_dist.minCoeff();
  FMCA::Scalar max_min_dist = min_dist.maxCoeff();
  std::cout << "fill distance:               " << max_min_dist << std::endl;
  std::cout << "separation distance:         " << min_min_dist << std::endl;
  T.tic();
  FMCA::ompSampletCompressor<H2SampletTree> comp;
  comp.init(hst, eta, threshold);
  T.toc("omp initializer:            ");
  comp.compress(hst, mat_eval);
  T.toc("cummulative compressor:     ");
  std::vector<Eigen::Triplet<double>> trips = comp.triplets();
  std::cout << "triplet size (\% of INT_MAX): "
            << (long double)(trips.size()) / (long double)(INT_MAX)*100
            << std::endl;
  FMCA::SparseMatrix<double>::sortTripletsInPlace(trips);
  double comperr =
      FMCA::errorEstimatorSymmetricCompressor(trips, function, hst, P);
  std::cout << "compression error:           " << comperr << std::endl
            << std::flush;
  T.tic();
  {
    FMCA::SparseMatrix<double> Sinput(npts, npts);
    Sinput.setFromTriplets(trips.begin(), trips.end());
    for (auto i = 0; i < Sinput.rows(); ++i)
      Sinput(i, i) = Sinput(i, i) + ridgep;
    trips = Sinput.toTriplets();
  }
  T.toc("added regularization:       ");
  // generate CRS matrix
  size_t n_triplets = trips.size();
  assert(n_triplets < INT_MAX && "exceeded INT_MAX");
  retval.ia.resize(npts + 1);
  retval.ja.resize(n_triplets);
  retval.a.resize(n_triplets);
  retval.ia[trips[0].row()] = 0;
  unsigned int j = 0;
  for (auto i = trips[0].row() + 1; i <= npts; ++i) {
    while (j < n_triplets && i - 1 == trips[j].row())
      ++j;
    retval.ia[i] = j;
  }
  assert(j == n_triplets && "j is not ntriplets");
  // write the rest
  for (auto i = 0; i < n_triplets; ++i) {
    retval.ja[i] = trips[i].col();
    retval.a[i] = trips[i].value();
  }
  return retval;
} // namespace FMCA
