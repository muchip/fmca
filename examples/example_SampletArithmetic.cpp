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
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
#include <FMCA/src/Samplets/samplet_matrix_multiplier.h>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>

struct expKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-(x - y).norm());
  }
};

using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, expKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main(int argc, char *argv[]) {
  const unsigned int dim = atoi(argv[1]);
  const unsigned int dtilde = 3;
  const auto function = expKernel();
  const double eta = 0.8;
  const unsigned int mp_deg = 4;
  const double threshold = 1e-5;
  FMCA::Tictoc T;
  for (unsigned int npts : {1e3, 5e3, 1e4, 5e4, 1e5}) {
    // for (unsigned int npts : {5e6}) {
    std::cout << "N:" << npts << " dim:" << dim << " eta:" << eta
              << " mpd:" << mp_deg << " dt:" << dtilde
              << " thres: " << threshold << std::endl;
    T.tic();
    const Eigen::MatrixXd P = Eigen::MatrixXd::Random(dim, npts);
    T.toc("geometry generation: ");
    const Moments mom(P, mp_deg);
    const MatrixEvaluator mat_eval(mom, function);
    const SampletMoments samp_mom(P, dtilde - 1);
    T.tic();
    H2SampletTree hst(mom, samp_mom, 0, P);
    T.toc("tree setup: ");
    FMCA::unsymmetric_compressor_impl<H2SampletTree> comp;
    T.tic();
    comp.compress(hst, mat_eval, eta, threshold);
    const double tcomp = T.toc("compressor: ");
    const auto &trips = comp.pattern_triplets();
    FMCA::SparseMatrix<double> S3(P.cols(), P.cols());
    FMCA::SparseMatrix<double> S4(P.cols(), P.cols());
    Eigen::SparseMatrix<double> S1(P.cols(), P.cols());
    Eigen::SparseMatrix<double> S2(P.cols(), P.cols());
    T.tic();
    S1.setFromTriplets(trips.begin(), trips.end());
    T.toc("eigen sparse: ");
    T.tic();
    S3.setFromTriplets(trips.begin(), trips.end());
    T.toc("FMCA sparse: ");

    S2 = S1;
    S4 = S3;
    T.tic();
    // Eigen::SparseMatrix<double> T1 = S1.transpose() * S2;
    T.toc("matrix product: ");
    T.tic();
    FMCA::samplet_matrix_multiplier<H2SampletTree> multip;
    multip.multiply(hst, S1, S2, eta, 1e-5);
    T.toc("matrix multiplier: ");
    T.tic();
    S3 *= S4;
    std::cout << S3.nnz() / P.cols() << std::endl;
    T.toc("fmca multiplier: ");
    const auto &trips2 = multip.pattern_triplets();
    // Eigen::SparseMatrix<double> T2(P.cols(), P.cols());
    // T2.setFromTriplets(trips2.begin(), trips2.end());
    // std::cout << "mult error: " << (T1 - T2).norm() / T1.norm() << std::endl;
    std::cout << "nzs: " << trips.size() / P.cols() << " "
              << trips2.size() / P.cols() << std::endl;
    std::cout << "------------------------------------------------------\n";
    std::cout << "------------------------------------------------------\n";
    // FMCA::IO::print2m("samp_mult.m", "T1", T1, "w");
    // FMCA::IO::print2m("samp_mult.m", "T2", T2, "a");
  }
  return 0;
}
