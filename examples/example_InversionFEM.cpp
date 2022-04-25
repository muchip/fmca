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
#include <FMCA/src/MatrixEvaluators/SparseMatrixEvaluator.h>

#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/Tictoc.h>

#include "pardiso_interface.h"
////////////////////////////////////////////////////////////////////////////////
using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SparseMatrixEvaluator = FMCA::SparseMatrixEvaluator<double>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const unsigned int dtilde = 4;
  const auto function = expKernel();
  const double eta = 0.5;
  const unsigned int mp_deg = 6;
  const double threshold = 0;
  FMCA::Tictoc T;
  const Eigen::MatrixXd P =
      readMatrix("../mex/mfiles/P_card_005.txt").transpose();
  const Eigen::MatrixXd Atrips = readMatrix("../mex/mfiles/A_card_005.txt");
  const unsigned int npts = P.cols();
  const unsigned int dim = P.rows();
  assert(Atrips(0, 0) == Atrips(0, 1) && Atrips(0, 0) == P.cols());
  std::cout << std::string(75, '=') << std::endl;
  std::cout << "npts:   " << npts << std::endl
            << "dim:    " << dim << std::endl
            << "dtilde: " << dtilde << std::endl
            << "mp_deg: " << mp_deg << std::endl
            << "eta:    " << eta << std::endl
            << std::flush;
  const Moments mom(P, mp_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("tree setup:        ");
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
  const SparseMatrixEvaluator mat_eval(A);
  FMCA::symmetric_compressor_impl<H2SampletTree> comp;
  T.tic();
  comp.compress(hst, mat_eval, eta, threshold);
  const double tcomp = T.toc("compressor:        ");
  T.tic();
  const auto &trips = comp.pattern_triplets();
  Eigen::SparseMatrix<double> S(npts, npts);
  Eigen::SparseMatrix<double> invS(npts, npts);
  FMCA::SparseMatrix<double> Sfmca(npts, npts);
  S.setFromTriplets(trips.begin(), trips.end());
  Sfmca.setFromTriplets(trips.begin(), trips.end());
  const auto sortTrips = Sfmca.toTriplets();
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
        inv_trips.push_back(Eigen::Triplet<double>(i, ja[j], a[j]));
    free(ia);
    free(ja);
    free(a);
    invS.setFromTriplets(inv_trips.begin(), inv_trips.end());
  }
  Eigen::MatrixXd rand = Eigen::MatrixXd::Random(npts, 100);
  auto Srand =
      S * rand + S.triangularView<Eigen::StrictlyUpper>().transpose() * rand;
  auto Rrand = invS * Srand +
               invS.triangularView<Eigen::StrictlyUpper>().transpose() * Srand;
  std::cout << "inverse error: " << (rand - Rrand).norm() / rand.norm()
            << std::endl;
  return 0;
}
