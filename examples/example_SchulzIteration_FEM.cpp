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
#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
#include <FMCA/src/MatrixEvaluators/SparseMatrixEvaluator.h>

////////////////////////////////////////////////////////////////////////////////
#include "../Points/matrixReader.h"
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>

using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SparseMatrixEvaluator = FMCA::SparseMatrixEvaluator<double>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main(int argc, char *argv[]) {
  const unsigned int dtilde = 3;
  const double eta = 0.8;
  const unsigned int mp_deg = 4;
  FMCA::Tictoc T;
  //////////////////////////////////////////////////////////////////////////////
  T.tic();
  std::cout << std::string(60, '=') << "\n";
  const Eigen::MatrixXd P =
      readMatrix("../mex/mfiles/P_mb_0001.txt").transpose();
  const Eigen::MatrixXd Atrips = readMatrix("../mex/mfiles/A_mb_0001.txt");
  T.toc("geometry generation: ");
  const unsigned int npts = P.cols();
  const unsigned int dim = P.rows();
  std::cout << std::string(60, '=') << "\n";
  std::cout << "N:" << npts << " dim:" << dim << " eta:" << eta
            << " mpd:" << mp_deg << " dt:" << dtilde << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  const Moments mom(P, mp_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("tree setup: ");
  assert(Atrips(0, 0) == Atrips(0, 1) && Atrips(0, 0) == P.cols());
  FMCA::SparseMatrix<double> A(Atrips(0, 0), Atrips(0, 1));
  FMCA::SparseMatrix<double> Asigma(Atrips(0, 0), Atrips(0, 1));
  FMCA::SparseMatrix<double> Pattern(Atrips(0, 0), Atrips(0, 1));
  //////////////////////////////////////////////////////////////////////////////
  std::vector<unsigned int> inv_idx(P.cols());
  std::vector<unsigned int> idx = hst.indices();
  double lambda_max = 0;
  {
    for (auto i = 0; i < idx.size(); ++i)
      inv_idx[idx[i]] = i;
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
  //////////////////////////////////////////////////////////////////////////////
  {
    const SparseMatrixEvaluator sparse_eval(A);
    FMCA::unsymmetric_compressor_impl<H2SampletTree> sp_comp;
    T.tic();
    sp_comp.compress(hst, sparse_eval, eta, 1e-14);
    T.toc("sparse_compressor: ");
    const auto &trips = sp_comp.pattern_triplets();
    Asigma.setFromTriplets(trips.begin(), trips.end());
    Asigma.compress(1e-12);
    Asigma.symmetrize();
    Pattern = sp_comp.pattern(hst, eta).full();
    Pattern.symmetrize();
  }
  FMCA::SparseMatrix<double> &S = Asigma;
  double trace = 0;
  for (auto i = 0; i < S.rows(); ++i)
    trace += S(i, i);
  std::cout << "trace: " << trace << " anz: " << S.nnz() / S.cols()
            << " anz: " << Pattern.nnz() / S.cols() << std::endl;
  std::cout << "norm: " << S.norm() << std::endl;
  //////////////////////////////////////////////////////////////////////////////
#if 0
  std::vector<const FMCA::TreeBase<H2SampletTree> *> leafs;
  for (auto level = 0; level < 16; ++level) {
    std::vector<Eigen::MatrixXd> bbvec;
    for (auto &node : hst) {
      if (node.level() == level) bbvec.push_back(node.derived().bb());
    }
    FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
  }
  FMCA::IO::plotPoints("points.vtk", P);
#endif
  //////////////////////////////////////////////////////////////////////////////
  FMCA::SparseMatrix<double> X(P.cols(), P.cols());
  FMCA::SparseMatrix<double> Xold(P.cols(), P.cols());
  FMCA::SparseMatrix<double> X0(P.cols(), P.cols());
  FMCA::SparseMatrix<double> Seps(P.cols(), P.cols());
  FMCA::SparseMatrix<double> ImXS(P.cols(), P.cols());
  FMCA::SparseMatrix<double> I2(P.cols(), P.cols());
  Eigen::MatrixXd randFilter = Eigen::MatrixXd::Random(S.rows(), 20);
  Eigen::VectorXd inv_diagS(S.rows());
  Seps = S;
  double alpha = 1. / lambda_max / lambda_max;
  std::cout << "chosen alpha for initial guess: " << alpha << std::endl;
  double err = 10;
  double err_old = 10;
  X = S;
  X.scale(alpha);
  std::cout << "initial guess: "
            << ((X * (S * randFilter)) - randFilter).norm() / randFilter.norm()
            << std::endl;
  I2.setIdentity().scale(2);
  for (auto inner_iter = 0; inner_iter < 100; ++inner_iter) {
    // ImXS = I2 - FMCA::SparseMatrix<double>::formatted_ABT(Pattern, X, Seps);
    // X = FMCA::SparseMatrix<double>::formatted_ABT(Pattern, X, ImXS);
    X = I2 * X - FMCA::SparseMatrix<double>::formatted_BABT(Pattern, Seps, X);
    std::cout << "apriori: " << X.nnz() / X.rows();
    X.compress(1e-8);
    X.symmetrize();
    std::cout << " aposteriori: " << X.nnz() / X.rows() << std::endl;
    err_old = err;
    err = ((X * (S * randFilter)) - randFilter).norm() / randFilter.norm();
    std::cout << "err: " << err << std::endl;
    if (err > err_old) {
      X = Xold;
      break;
    }
  }

  T.toc("time inner: ");
  std::cout << std::string(60, '=') << "\n";

  return 0;
}
