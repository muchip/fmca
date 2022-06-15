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
#include <fstream>
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/Sparse>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/SparseMatrix.h>

#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/HaltonSet.h>
#include <FMCA/src/util/Tictoc.h>
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
  /* d= 3, mp = 4, thresh = 1e-4 */
  /*npts dtilde mp_deg eta inv_eta filldist seprad tcomp comperr nnzS tmult
    nnzS2 nnzS2apost S2err*/
  typedef std::vector<Eigen::Triplet<double>> TripletVector;
  const unsigned int dtilde = 4;
  const double eta = atof(argv[2]);
  const unsigned int mp_deg = 6;
  const unsigned int dim = 2;
  const unsigned int npts = atoi(argv[1]);
  const auto function = expKernel(npts);
  const double threshold = 1e-5 / npts;
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
  output_file.open("output_Exp_" + std::to_string(dim) + "D.txt",
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
  TripletVector sym_trips = comp.pattern_triplets();
  TripletVector inv_trips = sym_trips;
  // TripletVector inv_trips = FMCA::symPattern(hst, inv_eta);
  FMCA::SparseMatrix<double>::sortTripletsInPlace(sym_trips);
  FMCA::SparseMatrix<double>::sortTripletsInPlace(inv_trips);
  FMCA::SparseMatrix<double> S(npts, npts);
  FMCA::SparseMatrix<double> Sp(npts, npts);
  FMCA::SparseMatrix<double> Spm1(npts, npts);
  FMCA::SparseMatrix<double> expS(npts, npts);
  Sp.setFromTriplets(inv_trips.begin(), inv_trips.end());
  S.setFromTriplets(sym_trips.begin(), sym_trips.end());
  S.mirrorUpper();
  double comperr =
      FMCA::errorEstimatorSymmetricCompressor(sym_trips, function, hst, P);
  std::cout << "compression error:  " << comperr << std::endl << std::flush;
  output_file << comperr << " \t";
  double lambda_max = 0;

  std::cout << "entries A:          "
            << 100 * double(sym_trips.size()) / npts / npts << "\%"
            << std::endl;
  output_file << 100 * double(sym_trips.size()) / npts / npts << " \t";
  std::cout << std::string(75, '=') << std::endl;
  T.tic();
  Eigen::MatrixXd rand = Eigen::MatrixXd::Random(npts, 100);
  Eigen::MatrixXd inc = S * rand;
  Eigen::MatrixXd expRand = rand + inc;
  expS.setIdentity();
  expS += S;
  Spm1 = Sp;
  Spm1.mirrorUpper();
  for (auto i = 2; i < 12; ++i) {
    inc = (1. / i) * (S * inc);
    expRand += inc;
    FMCA::SparseMatrix<double>::formatted_ABT(Sp, Spm1.scale(1. / i), S);
    std::cout << Sp.norm() << std::endl;
    Spm1 = Sp;
    Spm1.mirrorUpper();
    expS += Spm1;
  }
  const double tmult = T.toc("time mat mult:     ");

  Sp.mirrorUpper();
  std::cout << (expRand - expS * rand).norm() / rand.norm() << std::endl;
  output_file << tmult << " \t";
  output_file << 100 * double(Sp.nnz()) / npts / npts << " \t";

  double mult_err = 0;
  std::cout << "error:              " << mult_err << std::endl;
  output_file << mult_err << std::endl;
  output_file.close();
  return 0;
}
