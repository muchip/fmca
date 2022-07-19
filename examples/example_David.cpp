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
#include "formatted_multiplication.h"
#include "pardiso_interface.h"
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/Samplets/omp_samplet_compressor.h>
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/IO.h>
struct thinPCov {
  thinPCov(const double R) : R_(R), R3_(R * R * R) {}
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    const double r = (x - y).norm();
    return r * r * (2 * r - 3 * R_) + R3_;
  }
  const double R_;
  const double R3_;
};
////////////////////////////////////////////////////////////////////////////////
using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, thinPCov>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const unsigned int dtilde = 4;
  const double eta = 0.8;
  const unsigned int mp_deg = 6;
  const double ridgep = 1;
  double threshold = 1e-5;
  FMCA::Tictoc T;
  std::fstream output_file;
  Eigen::MatrixXd P;
  Eigen::MatrixXd Peval;
  Eigen::VectorXd y;
  {
    const unsigned int ninner = 1000;
    const unsigned int nouter = 1000;
    const unsigned int nbdry = 10000;
    const unsigned int neval = 20000;
    Eigen::MatrixXd Pts = readMatrix("../Points/David.txt");
    Eigen::MatrixXd mean = Pts.colwise().sum() / Pts.rows();
    std::cout << mean << std::endl;
    Pts = Pts - Eigen::MatrixXd::Ones(Pts.rows(), 1) * mean;
    const double scal = Pts.rowwise().norm().maxCoeff();
    std::cout << "geometry enc ball radius: " << scal << std::endl;
    Pts /= scal;
    std::vector<int> subsample(Pts.rows());
    std::iota(subsample.begin(), subsample.end(), 0);
    std::random_shuffle(subsample.begin(), subsample.end());
    P.resize(Pts.cols(), ninner + nouter + nbdry);
    y.resize(P.cols());
    for (auto i = 0; i < ninner; ++i) {
      Eigen::Vector3d rdm;
      rdm.setRandom();
      while (rdm.norm() > 0.1) rdm.setRandom();
      P.col(i) = rdm;
      y(i) = 1;
    }
    for (auto i = ninner; i < ninner + nbdry; ++i) {
      P.col(i) = Pts.row(subsample[i]).transpose();
      y(i) = 0;
    }
    for (auto i = ninner + nbdry; i < P.cols(); ++i) {
      Eigen::Vector3d rdm;
      rdm.setRandom();
      while (rdm.norm() < 0.6 || rdm.norm() > 0.9) rdm.setRandom();
      P.col(i) = 2 * rdm;
      y(i) = -1;
    }
    Peval.resize(P.rows(), neval);
    for (auto i = 0; i < Peval.cols(); ++i) {
      Eigen::Vector3d rdm;
      rdm.setRandom();
      while (rdm.norm() > 0.9) rdm.setRandom();
      Peval.col(i) = 2 * rdm;
    }
  }
  FMCA::IO::plotPointsColor("initalSetup.vtk", P, y);
  // P = Eigen::MatrixXd::Random(3, 10000);
  // Eigen::MatrixXd normP = P.colwise().norm();
  // for (auto i = 0; i < P.cols(); ++i) P.col(i) /= normP(i);
  //////////////////////////////////////////////////////////////////////////////
  const unsigned int npts = P.cols();
  const unsigned int dim = P.rows();
  const auto function = thinPCov(2);
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
  double comp_time = T.toc("cummulative compressor:     ");
  std::vector<Eigen::Triplet<double>> trips = comp.triplets();
  std::cout << "triplet size (\% of INT_MAX): "
            << (long double)(trips.size()) / (long double)(INT_MAX)*100
            << std::endl;
  std::cout << "anz:                         "
            << (long double)(trips.size()) / npts << std::endl;
  FMCA::SparseMatrix<double>::sortTripletsInPlace(trips);
  double comperr =
      FMCA::errorEstimatorSymmetricCompressor(trips, function, hst, P);
  std::cout << "compression error:           " << comperr << std::endl
            << std::flush;
  Eigen::SparseMatrix<double, Eigen::RowMajor> K(npts, npts);
  K.setFromTriplets(trips.begin(), trips.end());
  if (ridgep > 0) {
    T.tic();
    for (auto i = 0; i < K.rows(); ++i)
      K.coeffRef(i, i) = K.coeffRef(i, i) + ridgep;
    T.toc("added regularization:       ");
  }
  K.makeCompressed();
  Eigen::SparseMatrix<double, Eigen::RowMajor> invK = K;
  T.tic();
  pardiso_interface(invK.outerIndexPtr(), invK.innerIndexPtr(), invK.valuePtr(),
                    invK.rows());
  T.toc("matrix inversion:           ");
  double err = 0;
  {
    Eigen::MatrixXd x(npts, 10), y(npts, 10), z(npts, 10);
    x.setRandom();
    Eigen::VectorXd nrms = x.colwise().norm();
    for (auto i = 0; i < x.cols(); ++i) x.col(i) /= nrms(i);
    y.setZero();
    z.setZero();
    y = K.selfadjointView<Eigen::Upper>() * x;
    z = invK.selfadjointView<Eigen::Upper>() * y;
    err = (z - x).norm() / x.norm();
  }
  std::cout << "inverse error:               " << err << std::endl;
  auto I = hst.Indices();
  Eigen::VectorXd yI = y;
  Eigen::MatrixXd PI = P;
  for (auto i = 0; i < I.size(); ++i) yI(i) = y(I(i));
  for (auto i = 0; i < I.size(); ++i) PI.col(i) = P.col(I(i));
  Eigen::MatrixXd Keval(Peval.cols(), PI.cols());
  for (auto i = 0; i < PI.cols(); ++i)
    for (auto j = 0; j < Peval.cols(); ++j)
      Keval(j, i) = function(Peval.col(j), PI.col(i));

  return 0;
}
