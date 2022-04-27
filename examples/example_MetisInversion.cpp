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
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
using EigenCholesky =
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                         Eigen::MetisOrdering<int>>;
////////////////////////////////////////////////////////////////////////////////

#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>

#include "../Points/matrixReader.h"
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/MatrixEvaluators/SparseMatrixEvaluator.h>
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>
struct expKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-1 * (x - y).norm());
  }
};

using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, expKernel>;
using SparseMatrixEvaluator = FMCA::SparseMatrixEvaluator<double>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main(int argc, char *argv[]) {
  const unsigned int dtilde = 6;
  const auto function = expKernel();
  const double eta = 0.4;
  const unsigned int mp_deg = 6;
  const double threshold = 0;
  FMCA::Tictoc T;
#if 0
  const unsigned int dim = 2;
  const unsigned int npts = 1e1;
  T.tic();
  const Eigen::MatrixXd P = Eigen::MatrixXd::Random(dim, npts);
  T.toc("geometry generation: ");
#else
  T.tic();
  std::cout << std::string(60, '=') << "\n";
  const Eigen::MatrixXd P =
      readMatrix("../mex/mfiles/P_card_01.txt").transpose();
  const Eigen::MatrixXd Atrips = readMatrix("../mex/mfiles/A_card_01.txt");
  T.toc("geometry generation: ");
  const unsigned int npts = P.cols();
  const unsigned int dim = P.rows();
  std::cout << "N:" << npts << " dim:" << dim << " eta:" << eta
            << " mpd:" << mp_deg << " dt:" << dtilde << " thres: " << threshold
            << std::endl;
  assert(Atrips(0, 0) == Atrips(0, 1) && Atrips(0, 0) == P.cols());
  FMCA::SparseMatrix<double> A(Atrips(0, 0), Atrips(0, 1));
  for (auto i = 1; i < Atrips.rows(); ++i)
    A(Atrips(i, 0), Atrips(i, 1)) = Atrips(i, 2);
  FMCA::SparseMatrix<double> Pattern(Atrips(0, 0), Atrips(0, 1));
#endif
  const Moments mom(P, mp_deg);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("tree setup: ");
#if 0
  FMCA::symmetric_compressor_impl<H2SampletTree> comp;
  T.tic();
  comp.compress(hst, mat_eval, eta, threshold);
  const double tcomp = T.toc("compressor: ");
#else
  const SparseMatrixEvaluator sparse_eval(A);
  FMCA::unsymmetric_compressor_impl<H2SampletTree> comp;
  T.tic();
  comp.compress(hst, sparse_eval, eta, 1e-14);
  T.toc("sparse_compressor: ");
  Pattern = comp.pattern(hst, eta);
#endif
  const auto &trips = comp.pattern_triplets();
  Eigen::SparseMatrix<double> S(npts, npts);
  Eigen::SparseMatrix<double> iS(npts, npts);
  FMCA::SparseMatrix<double> invS(npts, npts);
  FMCA::SparseMatrix<double> Sfmca(npts, npts);
  S.setFromTriplets(trips.begin(), trips.end());
  Sfmca.setFromTriplets(trips.begin(), trips.end());
  const auto sortTrips = Sfmca.toTriplets();
  for (auto &&it : sortTrips)
    std::cout << it.row() << " " << it.col() << " " << it.value() << std::endl;
  invS = Pattern;
  EigenCholesky solver;
  T.tic();
  solver.compute(S);
  T.toc("time factorization: ");
  std::cout << "sinfo: " << solver.info() << std::endl;
  std::cout << "nz(L): "
            << solver.matrixL().nestedExpression().nonZeros() / P.cols()
            << std::endl;
  Eigen::VectorXd unit(npts);
  T.tic();
  for (auto i = 0; i < npts; ++i) {
    unit.setZero();
    unit(i) = 1;
    Eigen::VectorXd bla = solver.solve(unit);
    invS.setSparseRow(i, bla);
  }
  std::cout << invS.nnz() / npts << std::endl;
  invS.compress(1e-6);
  std::cout << invS.nnz() / npts << std::endl;
  T.toc("matrix inversion: ");
  const auto &inv_trips = invS.toTriplets();
  iS.setFromTriplets(inv_trips.begin(), inv_trips.end());
  Eigen::MatrixXd rand = Eigen::MatrixXd::Random(npts, 100);
  Eigen::MatrixXd y = S * rand;
  Eigen::MatrixXd y2 = iS * y;
  std::cout << (y2 - rand).norm() / rand.norm() << std::endl;
  return 0;
}
