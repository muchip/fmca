// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
#include <sys/time.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
////////////////////////////////////////////////////////////////////////////////
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/print2file.h"
#include "../Points/matrixReader.h"
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/Samplets/omp_samplet_compressor.h>
#include <FMCA/src/util/Errors.h>

#include "formatted_multiplication.h"
#include "pardiso_interface.h"
////////////////////////////////////////////////////////////////////////////////
struct expKernel {
  expKernel(const FMCA::Index n) : n_(n) {}
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    const double r = (x - y).norm();
    const double ell = 0.5;
    return 1. / n_ * exp(-r / ell);
    // return (1 + sqrt(3) * r / ell) * exp(-sqrt(3) * r / ell);
  }
  const FMCA::Index n_;
};
////////////////////////////////////////////////////////////////////////////////
using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, expKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const unsigned int dtilde = 4;
  const double eta = 0.8;
  const unsigned int mp_deg = 6;
  double threshold = 1e-5;
  double ridgep = 0;
  FMCA::Tictoc T;
  Eigen::MatrixXd P;
  const unsigned int n = atoi(argv[1]);
  //////////////////////////////////////////////////////////////////////////////
  // init data points
  //////////////////////////////////////////////////////////////////////////////
  {
    Eigen::MatrixXd Pts = readMatrix("../Points/bunnySurface.txt");
    Eigen::MatrixXd pts_min = Pts.colwise().minCoeff();
    Eigen::MatrixXd pts_max = Pts.colwise().maxCoeff();
    std::vector<int> subsample(Pts.rows());
    std::iota(subsample.begin(), subsample.end(), 0);
    std::random_shuffle(subsample.begin(), subsample.end());
    P.resize(Pts.cols(), n);
    for (auto i = 0; i < n; ++i) P.col(i) = Pts.row(subsample[i]).transpose();
  }
  //////////////////////////////////////////////////////////////////////////////
  // init samplet matrix
  //////////////////////////////////////////////////////////////////////////////
  const unsigned int npts = P.cols();
  const unsigned int dim = P.rows();
  std::fstream output_file;
  output_file.open("output_Exponential_FINAL_" + std::to_string(dim) + "D.txt",
                   std::ios::out | std::ios::app);
  const auto function = expKernel(npts);
  threshold /= npts;
  std::cout << std::string(75, '=') << std::endl;
  std::cout << "N:                           " << npts << std::endl
            << "dim:                         " << dim << std::endl
            << "eta:                         " << eta << std::endl
            << "multipole degree:            " << mp_deg << std::endl
            << "vanishing moments:           " << dtilde << std::endl
            << "aposteriori threshold:       " << threshold << std::endl
            << "ridge parameter:             " << ridgep << std::endl;
  output_file << "N:                           " << npts << std::endl
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
  output_file << "fill distance:               " << max_min_dist << std::endl;
  output_file << "separation distance:         " << min_min_dist << std::endl;
  T.tic();
  FMCA::ompSampletCompressor<H2SampletTree> comp;
  comp.init(hst, eta, threshold);
  comp.compress(hst, mat_eval);
  double comp_time = T.toc("cummulative compressor:     ");
  std::vector<Eigen::Triplet<double>> trips = comp.triplets();
  std::cout << "anz:                         " << (double)(trips.size()) / npts
            << std::endl;
  output_file << "anz:                         "
              << (double)(trips.size()) / npts << std::endl;

  FMCA::SparseMatrix<double>::sortTripletsInPlace(trips);
  double comperr =
      FMCA::errorEstimatorSymmetricCompressor(trips, function, hst, P);
  std::cout << "compression error:           " << comperr << std::endl;
  output_file << "compression error:           " << comperr << std::endl;
  Eigen::SparseMatrix<double, Eigen::RowMajor> K(npts, npts);
  K.setFromTriplets(trips.begin(), trips.end());
  if (ridgep > 0) {
    T.tic();
    for (auto i = 0; i < K.rows(); ++i)
      K.coeffRef(i, i) = K.coeffRef(i, i) + ridgep;
    T.toc("added regularization:       ");
  }
  K.makeCompressed();
  double op_err = 0;
  {
    Eigen::VectorXd x = Eigen::VectorXd::Random(K.cols());
    x /= x.norm();
    for (auto i = 0; i < 50; ++i) {
      x = K.selfadjointView<Eigen::Upper>() * x;
      op_err = x.norm();
      x /= op_err;
    }
    std::cout << "op. norm err (50its of power it): " << op_err << std::endl;
    output_file << "op. norm err (50its of power it): " << op_err << std::endl;
  }
  //////////////////////////////////////////////////////////////////////////////
  // perform power series for exponential
  //////////////////////////////////////////////////////////////////////////////
  largeSparse Kpk(npts, npts);
  largeSparse Kpkm1(npts, npts);
  Eigen::SparseMatrix<double, Eigen::RowMajor> bla =
      K.selfadjointView<Eigen::Upper>();
  largeSparse Ksym = bla;
  largeSparse expK(npts, npts);
  expK.setIdentity();
  expK += Ksym;
  Kpk = Ksym;
  Kpkm1 = Kpk;
  Eigen::MatrixXd rdm = Eigen::MatrixXd::Random(npts, 10);
  Eigen::MatrixXd Kprdm = Ksym * rdm;
  Eigen::MatrixXd expKrdm = rdm + Kprdm;
  for (auto i = 2; i < 30; ++i) {
    Kprdm = (1. / i) * Ksym * Kprdm;
    expKrdm += Kprdm;
  }

  for (auto i = 2; i <= 10; ++i) {
    output_file << "i= " << i << " ";
    memset(Kpk.valuePtr(), 0, Kpk.nonZeros() * sizeof(double));
    T.tic();
    formatted_sparse_multiplication(Kpk, Ksym, Kpkm1, 1. / i);
    Kpkm1 = Kpk;
    double time_mult = T.toc();
    expK += Kpk;
    double err = (expKrdm - expK * rdm).norm() / expKrdm.norm();
    std::cout << "exponential error: " << err << std::endl;
    output_file << " t_mult: " << time_mult << " err: " << err << std::endl;
  }
  output_file.close();
  return 0;
}
