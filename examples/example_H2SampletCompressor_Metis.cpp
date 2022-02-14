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
//
#define FMCA_CLUSTERSET_
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
using EigenCholesky =
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                         Eigen::MetisOrdering<int>>;

#include <FMCA/Samplets>

#include "../FMCA/src/util/Errors.h"
#include "../FMCA/src/util/NormalDistribution.h"
#include "../FMCA/src/util/print2file.hpp"
#include "../FMCA/src/util/tictoc.hpp"
#include "../Points/matrixReader.h"
#include "generateSwissCheese.h"
#include "generateSwissCheeseExp.h"

struct exponentialKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-(x - y).norm());
  }
};

// struct rationalQuadraticKernel {
//  template <typename derived, typename otherDerived>
//  double operator()(const Eigen::MatrixBase<derived> &x,
//                    const Eigen::MatrixBase<otherDerived> &y) const {
//    const double r = (x - y).norm();
//    constexpr double alpha = 0.5;
//    constexpr double ell = 1.;
//    constexpr double c = 1. / (2. * alpha * ell * ell);
//    return std::pow(1 + c * r * r, -alpha);
//  }
//};

struct rationalQuadraticKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    const double r = (x - y).norm();
    constexpr double alpha = 0.5;
    constexpr double ell = 1.;
    constexpr double c = 1. / (2. * alpha * ell * ell);
    return 1. / sqrt(1 + c * r * r);
  }
};

using theKernel = exponentialKernel;

const double parameters[4][3] = {
    {2, 1, 1e-2}, {3, 2, 1e-3}, {4, 3, 1e-4}, {6, 4, 1e-5}};

int main(int argc, char *argv[]) {
  constexpr double ridge_param = 1e-12;
  constexpr unsigned int dim = 4;
  constexpr unsigned int dtilde = 3;
  const auto function = theKernel();
  const double eta = 0.8;
  const unsigned int mp_deg = parameters[dtilde - 1][0];
  const double threshold = parameters[dtilde - 1][2];
  tictoc T;
  std::fstream file;

  Eigen::MatrixXd theP;
  Eigen::MatrixXd Pts = readMatrix("../../Points/bunnyFine.txt");
  if (4 == dim) {
    Eigen::MatrixXd Q(4, 4);
    Q << -0.174461059412466, 0.262091364616281, -0.730726716121948,
        -0.605730898739606, -0.595055005141286, -0.790441432527659,
        -0.145227429801521, 0.004569051670754, 0.732945697135043,
        -0.464176309187930, -0.471814564843441, 0.157232234104495,
        -0.279756116387487, 0.301746271436564, -0.471527810823999,
        0.779966170187816;
    theP.resize(4, Pts.rows());
    theP.setZero();
    theP.topRows(3) = Pts.transpose();
    theP = Q * theP;
  }
  file.open("s_output" + std::to_string(dim) + "_" + std::to_string(dtilde) +
                "_Metis.txt",
            std::ios::out);

  file << "         n       nz(A)         mem          err       time        "
          " cerr      ctime       sinfo       nz(L)\n";
  for (unsigned int npts : {1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6}) {
    std::cout << "N:" << npts << " dim:" << dim << " eta:" << eta
              << " mpd:" << mp_deg << " dt:" << dtilde
              << " thres: " << threshold << std::endl;
    T.tic();
    const Eigen::MatrixXd P = theP.leftCols(npts);
    T.toc("geometry generation: ");

    const FMCA::NystromMatrixEvaluator<FMCA::H2SampletTree, theKernel> nm_eval(
        P, function);
    T.tic();
    FMCA::H2SampletTree ST(P, 10, dtilde, mp_deg);
    T.toc("tree setup: ");
    FMCA::symmetric_compressor_impl<FMCA::H2SampletTree> symComp;
    T.tic();
    symComp.compress(ST, nm_eval, eta, threshold);
    double tcomp = T.toc("symmetric compressor: ");
    {
      Eigen::SparseMatrix<double> Ssym(npts, npts);
      Eigen::VectorXd x(npts), y1(npts), y2(npts);
      double err = 0;
      double nrm = 0;
      const auto &trips = symComp.pattern_triplets();
      file << std::setw(10) << std::setprecision(6) << npts << "\t";
      file << std::setw(10) << std::setprecision(6)
           << std::ceil(double(trips.size()) / npts) << "\t";
      file << std::flush;
      Ssym.setFromTriplets(trips.begin(), trips.end());
      double trace = Ssym.diagonal().sum();
      std::cout << "trace: " << trace << std::endl;
      file << std::setw(10) << std::setprecision(5)
           << 3 * double(trips.size()) * sizeof(double) / 1e9 << "\t";
      std::cout << "nz(S): " << std::ceil(double(trips.size()) / npts)
                << std::endl;
      std::cout << "memory: " << 3 * double(trips.size()) * sizeof(double) / 1e9
                << "GB\n"
                << std::flush;
      for (auto i = 0; i < 100; ++i) {
        unsigned int index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        y1 = FMCA::matrixColumnGetter(P, ST.indices(), function, index);
        x = ST.sampletTransform(x);
        y2 = Ssym * x +
             Ssym.triangularView<Eigen::StrictlyUpper>().transpose() * x;
        y2 = ST.inverseSampletTransform(y2);
        err += (y1 - y2).squaredNorm();
        nrm += y1.squaredNorm();
      }
      err = sqrt(err / nrm);
      std::cout << "compression error: " << err << std::endl;
      if (1e5 == npts) Bembel::IO::print2spascii("bunnyMat.txt", Ssym, "w");
      file << std::setw(10) << std::setprecision(6) << err << "\t";
      file << std::setw(10) << std::setprecision(6) << tcomp << "\t";
      file << std::flush;
      std::cout << "starting Cholesky decomposition\n";
      Eigen::SparseMatrix<double> I(npts, npts);
      I.setIdentity();
      Ssym += ridge_param * trace * I;
      EigenCholesky solver;
      T.tic();
      solver.compute(Ssym);
      tcomp = T.toc("time factorization: ");
      std::cout << "sinfo: " << solver.info() << std::endl;
      std::cout << "nz(L): "
                << solver.matrixL().nestedExpression().nonZeros() / P.cols()
                << std::endl;
      if (1e5 == npts) Bembel::IO::print2spascii("bunnyChol.txt", solver.matrixL().nestedExpression(), "w");
      err = 0;
      nrm = 0;
      for (auto i = 0; i < 10; ++i) {
        x.setRandom();
        y1 = solver.permutationPinv() *
             (solver.matrixL() * (solver.matrixL().transpose() *
                                  (solver.permutationP() * x).eval())
                                     .eval())
                 .eval();
        y2 = Ssym * x +
             Ssym.triangularView<Eigen::StrictlyUpper>().transpose() * x;
        err += (y1 - y2).squaredNorm();
        nrm += y2.squaredNorm();
      }
      err = sqrt(err / nrm);
      std::cout << "Cholesky error: " << err << std::endl;
      file << std::setw(10) << std::setprecision(6) << err << "\t";
      file << std::setw(10) << std::setprecision(6) << tcomp << "\t";
      file << std::setw(10) << std::setprecision(6) << solver.info() << "\t";
      file << std::setw(10) << std::setprecision(6)
           << solver.matrixL().nestedExpression().nonZeros() / P.cols() << "\n";
      file << std::flush;

      FMCA::NormalDistribution ND(0, 1, 0);
      Eigen::VectorXd data;
      for (auto i = 0; i < 20; ++i) {
        data = ND.get_randMat(P.cols(), 1);
        data = ST.sampletTransform(data);
        data = solver.permutationPinv() * (solver.matrixL() * data).eval();
        data = ST.inverseSampletTransform(data);
        FMCA::IO::plotPoints("points" + std::to_string(i) + ".vtk", ST,
                             Pts.topRows(npts).transpose(), data);
      }
    }
    std::cout << std::string(60, '-') << std::endl;
  }
  file.close();
  return 0;
}

#if 0
  std::vector<unsigned int> perm(Pts.rows());
  std::iota(perm.begin(), perm.end(), 0);
  std::random_shuffle(perm.begin(), perm.end());
  Eigen::MatrixXd Pts2(Pts.rows(), Pts.cols());
  for (auto i = 0; i < Pts2.rows(); ++i) Pts2.row(i) = Pts.row(perm[i]);
  std::fstream ofile;
  ofile.open("permBunny.txt", std::ios::out);
  for (auto i = 0; i < Pts2.rows(); ++i)
    ofile << std::setprecision(8) << Pts(i, 0) << " " << Pts(i, 1) << " "
          << Pts(i, 2) << std::endl;
  ofile.close();
  Eigen::MatrixXd P = Pts2.topRows(1e6).transpose();
  FMCA::IO::plotPoints("points.vtk", P);
  return 0;
#endif
