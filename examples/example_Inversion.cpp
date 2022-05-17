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
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
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
#include <FMCA/src/util/print2file.h>

#include "pardiso_interface.h"
////////////////////////////////////////////////////////////////////////////////
struct expKernel {
  expKernel(const unsigned int n) : n_(n) {}
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    const double r = (x - y).norm();
    const double ell = 1;
    return 1. / n_ * exp(-r / ell);
    // return (1 + sqrt(3) * r / ell) * exp(-sqrt(3) * r / ell);
  }
  const unsigned int n_;
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
  /* d= 4, mp = 6, thresh = 1e-5 */
  /*npts dtilde mp_deg eta filldist seprad tcomp comperr nnzS tmult
    nnzS2 nnzS2apost S2err*/
  typedef std::vector<Eigen::Triplet<double>> TripletVector;
  const unsigned int dtilde = 4;
  const double eta = atof(argv[2]);
  const unsigned int mp_deg = 6;
  const unsigned int dim = 3;
  const unsigned int npts = atoi(argv[1]);
  const double threshold = 1e-5 / npts;
  const auto function = expKernel(npts);
  std::fstream output_file;
  FMCA::HaltonSet<100> hs(dim);
  FMCA::Tictoc T;
  Eigen::MatrixXd P = 0.5 * (Eigen::MatrixXd::Random(dim, npts).array() + 1);
#if 0
  for (auto i = 0; i < P.cols(); ++i) {
    P.col(i) = hs.EigenHaltonVector();
    hs.next();
  }
#endif
  output_file.open("output_Inversion_" + std::to_string(dim) + "D.txt",
                   std::ios::out | std::ios::app);
  std::cout << std::string(75, '=') << std::endl;
  std::cout << "npts: " << npts << " | dim: " << dim << " | dtilde: " << dtilde
            << " | mp_deg: " << mp_deg << " | eta: " << eta << std::endl
            << std::flush;
  output_file << npts << " \t" << dtilde << " \t" << mp_deg << " \t" << eta
              << " \t";
  const Moments mom(P, mp_deg);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("tree setup:        ");
  double filldist = FMCA::fillDistance(hst, P);
  double seprad = FMCA::separationRadius(hst, P);
  std::cout << "fill_dist: " << filldist << std::endl;
  std::cout << "sep_rad: " << seprad << std::endl;
  std::cout << "bb: " << std::endl << hst.bb() << std::endl;
  FMCA::symmetric_compressor_impl<H2SampletTree> comp;
  T.tic();
  comp.compress(hst, mat_eval, eta, threshold);
  const double tcomp = T.toc("compressor:        ");
  output_file << filldist << " \t" << seprad << " \t" << tcomp << " \t";
  std::vector<Eigen::Triplet<double>> trips = comp.pattern_triplets();
    std::cout << "triplet size (\% of INT_MAX):"
              << (long double)(trips.size()) / (long double)(INT_MAX)*100
              << std::endl;
  std::vector<Eigen::Triplet<double>> inv_trips;
  FMCA::SparseMatrix<double>::sortTripletsInPlace(trips);
  Eigen::MatrixXd mtrips(trips.size() + 1, 3);
  Eigen::MatrixXd read_trips;

  double comperr =
      FMCA::errorEstimatorSymmetricCompressor(trips, function, hst, P);
  std::cout << "compression error:  " << comperr << std::endl << std::flush;
  output_file << comperr << " \t";
  T.tic();
  {
    FMCA::SparseMatrix<double> Sinput(npts, npts);
    Sinput.setFromTriplets(trips.begin(), trips.end());
    for (auto i = 0; i < Sinput.rows(); ++i)
      Sinput(i, i) = Sinput(i, i) + 1e-4;
    trips = Sinput.toTriplets();
  }
  T.toc("added regularization: ");
  T.tic();
  mtrips.setZero();
  {
    mtrips.row(0) << npts, npts, 0;
    for (auto i = 1; i < mtrips.rows(); ++i)
      mtrips.row(i) << trips[i - 1].row(), trips[i - 1].col(),
          trips[i - 1].value();
    FMCA::IO::print2bin("matrix" + std::to_string(npts) + ".dat", mtrips);
    FMCA::IO::bin2Mat("matrix" + std::to_string(npts) + ".dat", &read_trips);
    std::cout << (mtrips - read_trips).colwise().maxCoeff() << std::endl;
  }
  T.toc("export: ");
  return 0;
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
  output_file << lambda_max << " \t" << std::flush;

  std::cout << "entries A:          "
            << 100 * double(trips.size()) / npts / npts << "\%" << std::endl;
  std::cout << std::string(75, '=') << std::endl;
  output_file << 100 * double(trips.size()) / npts / npts << " \t"
              << std::flush;
  {
    int i = 0;
    int j = 0;
    int n = npts;
    size_t n_triplets = trips.size();
    std::cout << "triplet size (\% of INT_MAX):"
              << (long double)(n_triplets) / (long double)(INT_MAX)*100
              << std::endl;
    assert(n_triplets < INT_MAX && "exceeded INT_MAX");
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
      while (j < n_triplets && i - 1 == trips[j].row())
        ++j;
      ia[i] = j;
    }
    assert(j == n_triplets && "j is not ntriplets");
    // write the rest
    for (i = 0; i < n_triplets; ++i) {
      ja[i] = trips[i].col();
      a[i] = trips[i].value();
    }
    std::cout << "\n\nentering pardiso block\n" << std::flush;
    T.tic();
    std::printf("ia=%p ja=%p a=%p n=%i nnz=%i\n", ia, ja, a, n, ia[n]);
    std::cout << std::flush;
    pardiso_interface(ia, ja, a, n);
    std::cout << std::string(75, '=') << std::endl;
    const double tPard = T.toc("Wall time pardiso: ");
    output_file << tPard << " \t";
    inv_trips.reserve(n_triplets);
    for (i = 0; i < n; ++i)
      for (j = ia[i]; j < ia[i + 1]; ++j)
        if (abs(a[j]) > 1e-12)
          inv_trips.push_back(Eigen::Triplet<double>(i, ja[j], a[j]));
    free(ia);
    free(ja);
    free(a);
  }
  std::cout << "inverse entries:    " << 100. * inv_trips.size() / npts / npts
            << "\%" << std::endl;
  output_file << 100. * inv_trips.size() / npts / npts << " \t";
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
    output_file << (rand - Rrand).norm() / rand.norm() << " \t";
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
    output_file << lambda_max << std::endl;
  }
  output_file.close();
  return 0;
}
