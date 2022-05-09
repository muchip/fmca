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
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-1 * (x - y).norm());
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
  /*npts dtilde mp_deg eta inv_eta filldist seprad tcomp comperr nnzS tmult
    nnzS2 nnzS2apost S2err*/
  typedef std::vector<Eigen::Triplet<double>> TripletVector;
  const unsigned int dtilde = 3;
  const auto function = expKernel();
  const double eta = atof(argv[2]);
  const double inv_eta = atof(argv[3]);
  const unsigned int mp_deg = 4;
  const double threshold = 1e-4;
  const double threshold2 = 1e-12;
  const unsigned int dim = 3;
  const unsigned int npts = atoi(argv[1]);
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
  output_file.open("output_PAT_" + std::to_string(dim) + "D.txt",
                   std::ios::out | std::ios::app);
  std::cout << std::string(75, '=') << std::endl;
  std::cout << "npts: " << npts << " | dim: " << dim << " | dtilde: " << dtilde
            << " | mp_deg: " << mp_deg << " | eta: " << eta
            << " | inv_eta: " << inv_eta << std::endl
            << std::flush;
  output_file << npts << " \t" << dtilde << " \t" << mp_deg << " \t" << eta
              << " \t" << inv_eta << " \t";
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
  //TripletVector inv_trips = FMCA::symPattern(hst, inv_eta);
  FMCA::SparseMatrix<double>::sortTripletsInPlace(sym_trips);
  FMCA::SparseMatrix<double>::sortTripletsInPlace(inv_trips);
  FMCA::SparseMatrix<double> S(npts, npts);
  FMCA::SparseMatrix<double> S2(npts, npts);
  S2.setFromTriplets(inv_trips.begin(), inv_trips.end());
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
  for (auto i = 0; i < 1; ++i) {
    FMCA::SparseMatrix<double>::formatted_ABT(S2, S, S);
  }
  const double tmult = T.toc("time mat mult:     ");
  output_file << tmult << " \t";
  output_file << 100 * double(S2.nnz()) / npts / npts << " \t";


  S2.compress(threshold2);
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
