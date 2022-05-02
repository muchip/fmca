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
    const double ell = 0.2;
    return exp(-r / ell);
    // return (1 + sqrt(3) * r / ell) * exp(-sqrt(3) * r / ell);
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
  const unsigned int mp_deg = 6;
  const double threshold = -1;
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
            << " | mp_deg: " << mp_deg << " | eta: " << eta << std::endl
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
  double nz = 0;
  for (auto &&it : trips)
    if (abs(it.value()) < 1e-8)
      it = Eigen::Triplet<double>(it.row(), it.col(), 0);
    else
      nz += 1;
  std::cout << "nnz: " << nz / npts / npts * 100. << std::endl;
  T.tic();
  Eigen::VectorXd x(npts), y1(npts), y2(npts);
  double err = 0;
  double nrm = 0;
  for (auto i = 0; i < 100; ++i) {
    unsigned int index = rand() % P.cols();
    x.setZero();
    x(index) = 1;
    y1 = FMCA::matrixColumnGetter(P, hst.indices(), function, index);
    x = hst.sampletTransform(x);
    y2.setZero();
    for (const auto &i : trips) {
      y2(i.row()) += i.value() * x(i.col());
      if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
    }
    y2 = hst.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }

  const double thet = T.toc("matrix vector time: ");
  std::cout << "average matrix vector time " << thet / 100 << "sec."
            << std::endl;
  err = sqrt(err / nrm);
  std::cout << "compression error: " << err << std::endl << std::flush;
  Eigen::SparseMatrix<double> S(npts, npts);
  Eigen::SparseMatrix<double> invS(npts, npts);
  FMCA::SparseMatrix<double> Sfmca(npts, npts);
  FMCA::SparseMatrix<double> D(npts, npts);

  Sfmca.setFromTriplets(trips.begin(), trips.end());
  S.setFromTriplets(trips.begin(), trips.end());
  double lambda_max = 0;
  {
    Eigen::MatrixXd x = Eigen::VectorXd::Random(S.cols());
    x /= x.norm();
    for (auto i = 0; i < 20; ++i) {
      x = S * x + S.triangularView<Eigen::StrictlyUpper>().transpose() * x;
      lambda_max = x.norm();
      x /= lambda_max;
    }
    std::cout << "lambda_max (est by 20its of power it): " << lambda_max
              << std::endl;
  }
  std::cout << "entries A:          " << 100 * double(Sfmca.nnz()) / npts / npts
            << "\%" << std::endl;
  double trace = 0;
  for (auto i = 0; i < Sfmca.cols(); ++i) trace += Sfmca(i, i);
  std::cout << trace << std::endl;

  for (auto i = 0; i < Sfmca.cols(); ++i)
    D(i, i) = 1. / sqrt(Sfmca(i, i) + 1e-4);
  Sfmca = D * (Sfmca * D);
  for (auto i = 0; i < Sfmca.cols(); ++i) Sfmca(i, i) += 1e-4;

  double trace2 = 0;
  for (auto i = 0; i < Sfmca.cols(); ++i) trace2 += Sfmca(i, i);
  std::cout << trace2 << " ratio: " << trace2 / trace - 1 << std::endl;

  const auto sortTrips = Sfmca.toTriplets();
  S.setFromTriplets(sortTrips.begin(), sortTrips.end());
  T.toc("sparse matrices:   ");
  std::cout << std::string(75, '=') << std::endl;
  {
    int i = 0;
    int j = 0;
    int n = Sfmca.cols();
    int m = Sfmca.rows();
    int n_triplets = sortTrips.size();
    int *ia = nullptr;
    int *ja = nullptr;
    double *a = nullptr;
    ia = (int *)malloc((m + 1) * sizeof(int));
    ja = (int *)malloc(n_triplets * sizeof(int));
    a = (double *)malloc(n_triplets * sizeof(double));
    memset(ia, 0, (m + 1) * sizeof(int));
    memset(ja, 0, n_triplets * sizeof(int));
    memset(a, 0, n_triplets * sizeof(double));
    // write rows
    ia[sortTrips[0].row()] = 0;
    for (i = sortTrips[0].row() + 1; i <= m; ++i) {
      while (j < n_triplets && i - 1 == sortTrips[j].row()) ++j;
      ia[i] = j;
    }
    // write the rest
    for (i = 0; i < n_triplets; ++i) {
      ja[i] = sortTrips[i].col();
      a[i] = sortTrips[i].value();
    }
    std::cout << "\n\nentering pardiso block" << std::flush;
    T.tic();
    pardiso_interface(ia, ja, a, m, n);
    std::cout << std::string(75, '=') << std::endl;
    T.toc("Wall time pardiso: ");
    std::vector<Eigen::Triplet<double>> inv_trips;
    for (i = 0; i < m; ++i)
      for (j = ia[i]; j < ia[i + 1]; ++j)
        if (abs(a[j]) > 1e-8)
          inv_trips.push_back(Eigen::Triplet<double>(i, ja[j], a[j]));
    free(ia);
    free(ja);
    free(a);
    invS.setFromTriplets(inv_trips.begin(), inv_trips.end());
  }
  std::cout << "inverse entries: " << 100. * invS.nonZeros() / npts / npts
            << std::endl;
  {
    Eigen::MatrixXd rand = Eigen::MatrixXd::Random(npts, 100);
    Eigen::VectorXd nrms = rand.colwise().norm();
    for (auto i = 0; i < rand.cols(); ++i) rand.col(i) /= nrms(i);
    auto Srand =
        S * rand + S.triangularView<Eigen::StrictlyUpper>().transpose() * rand;
    auto Rrand =
        invS * Srand +
        invS.triangularView<Eigen::StrictlyUpper>().transpose() * Srand;
    std::cout << "inverse error: " << (rand - Rrand).norm() / rand.norm() << " "
              << rand.norm() << std::endl;
  }
  {
    Eigen::VectorXd x = Eigen::VectorXd::Random(S.cols());
    Eigen::VectorXd xold;
    Eigen::VectorXd y;
    x /= x.norm();
    for (auto i = 0; i < 50; ++i) {
      xold = x;
      y = S.triangularView<Eigen::Upper>() * xold +
          S.triangularView<Eigen::StrictlyUpper>().transpose() * xold;
      x = invS.triangularView<Eigen::Upper>() * y +
          invS.triangularView<Eigen::StrictlyUpper>().transpose() * y;
      x -= xold;
      lambda_max = x.norm();
      x /= lambda_max;
    }
    std::cout << "op. norm err (50its of power it): " << lambda_max
              << std::endl;
  }

  return 0;
}
