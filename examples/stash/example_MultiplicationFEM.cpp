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
#include <FMCA/src/MatrixEvaluators/SparseMatrixEvaluator.h>
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/HaltonSet.h>
#include <FMCA/src/util/Tictoc.h>
////////////////////////////////////////////////////////////////////////////////
#include "../Points/matrixReader.h"
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
using SparseMatrixEvaluator = FMCA::SparseMatrixEvaluator<double>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  /* d= 4, mp = 6, thresh = 1e-5 */
  /* d= 3, mp = 4, thresh = 1e-4 */
  /*npts dtilde mp_deg eta inv_eta filldist seprad tcomp comperr nnzS tmult
    nnzS2 nnzS2apost S2err*/
  typedef std::vector<Eigen::Triplet<double>> TripletVector;
  const unsigned int dtilde = 1;
  const double eta = 0.8;
  const unsigned int mp_deg = 2;
  const double threshold = 1e-10;
  std::fstream output_file;
  FMCA::Tictoc T;
  const Eigen::MatrixXd P =
      readMatrix("../mex/mfiles/P_lshape_001.txt").transpose();
  const Eigen::MatrixXd Atrips = readMatrix("../mex/mfiles/A_lshape_001.txt");
  const unsigned int npts = P.cols();
  const unsigned int dim = P.rows();
  assert(Atrips(0, 0) == Atrips(0, 1) && Atrips(0, 0) == P.cols());
#if 0
  for (auto i = 0; i < P.cols(); ++i) {
    P.col(i) = hs.EigenHaltonVector();
    hs.next();
  }
#endif
  output_file.open("output_PATFE_" + std::to_string(dim) + "D.txt",
                   std::ios::out | std::ios::app);
  std::cout << std::string(75, '=') << std::endl;
  std::cout << "npts: " << npts << " | dim: " << dim << " | dtilde: " << dtilde
            << " | mp_deg: " << mp_deg << " | eta: " << eta << std::endl
            << std::flush;
  output_file << npts << " \t" << dtilde << " \t" << mp_deg << " \t" << eta
              << " \t";
  const Moments mom(P, mp_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("tree setup:        ");
  double filldist = FMCA::fillDistance(hst, P);
  double seprad = FMCA::separationRadius(hst, P);
  std::cout << "fill_dist: " << filldist << std::endl;
  std::cout << "sep_rad: " << seprad << std::endl;
  std::cout << "bb: " << std::endl << hst.bb() << std::endl;
  FMCA::SparseMatrix<double> A(npts, npts);
  //////////////////////////////////////////////////////////////////////////////
  std::vector<unsigned int> inv_idx(P.cols());
  std::vector<unsigned int> idx = hst.indices();
  double lambda_max = 0;
  {
    for (auto i = 0; i < idx.size(); ++i) inv_idx[idx[i]] = i;
    for (auto i = 1; i < Atrips.rows(); ++i) {
      A(Atrips(i, 0), Atrips(i, 1)) = Atrips(i, 2);
    }
    Eigen::MatrixXd x = Eigen::VectorXd::Random(A.cols());
    x /= x.norm();
    for (auto i = 0; i < 20; ++i) {
      x = A * x;
      lambda_max = x.norm();
      x /= lambda_max;
    }
    std::cout << "lambda_max (est by 20its of power it): " << lambda_max
              << std::endl;
  }
  std::cout << "domain diam: " << hst.bb().col(2).norm() << std::endl;
  const SparseMatrixEvaluator mat_eval(A);
  
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
  FMCA::SparseMatrix<double> S2(npts, npts);
  S2.setFromTriplets(inv_trips.begin(), inv_trips.end());
  S.setFromTriplets(sym_trips.begin(), sym_trips.end());
  S.mirrorUpper();

  std::cout << "entries A:          "
            << 100 * double(sym_trips.size()) / npts / npts << "\%"
            << std::endl;
  output_file << 100 * double(sym_trips.size()) / npts / npts << " \t";
  std::cout << std::string(75, '=') << std::endl;
  T.tic();
  for (auto i = 0; i < 1; ++i) {
    FMCA::SparseMatrix<double>::formatted_ABT(S2, S, S);
  }
  const double tmult = T.toc("time mat mult:     ");
  output_file << tmult << " \t";
  output_file << 100 * double(S2.nnz()) / npts / npts << " \t";

  Eigen::MatrixXd rand = Eigen::MatrixXd::Random(npts, 100);
  auto Srand = S * (S * rand);
  inv_trips = S2.toTriplets();
  output_file << 100 * double(inv_trips.size()) / npts / npts << " \t";

  auto Rrand =
      FMCA::SparseMatrix<double>::symTripletsTimesVector(inv_trips, rand);
  double mult_err = (Srand - Rrand).norm() / Srand.norm();
  std::cout << "error:              " << mult_err << std::endl;
  output_file << mult_err << std::endl;
  output_file.close();
  return 0;
}
