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
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/Tictoc.h>

struct expKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-(x - y).norm()) * x.norm();
  }
};

using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator =
    FMCA::unsymmetricNystromMatrixEvaluator<Moments, expKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main(int argc, char *argv[]) {
  const unsigned int dim = atoi(argv[1]);
  const unsigned int dtildeR = 3;
  const unsigned int dtildeC = 4;
  const auto function = expKernel();
  const double eta = 0.5;
  const unsigned int mp_degR = 4;
  const unsigned int mp_degC = 6;
  const double threshold = 1e-10;
  const unsigned int mpts = 10000;
  const unsigned int npts = 5000;

  FMCA::Tictoc T;
  std::cout << "M:" << mpts << " N: " << npts << " dim:" << dim
            << " eta:" << eta << " mpd:" << mp_degR << " dt:" << dtildeR
            << " thres: " << threshold << std::endl;
  T.tic();
  const Eigen::MatrixXd PR = Eigen::MatrixXd::Random(dim, mpts);
  const Eigen::MatrixXd PC = Eigen::MatrixXd::Random(dim, npts);
  T.toc("geometry generation: ");
  const Moments momR(PR, mp_degR);
  const Moments momC(PC, mp_degC);
  const MatrixEvaluator mat_eval(momR, momC, function);
  const SampletMoments samp_momR(PR, dtildeR - 1);
  const SampletMoments samp_momC(PC, dtildeC - 1);
  T.tic();
  H2SampletTree hstR(momR, samp_momR, 0, PR);
  H2SampletTree hstC(momC, samp_momC, 0, PC);
  T.toc("tree setup: ");
  std::cout << std::flush;
  Eigen::MatrixXd block;
  mat_eval.compute_dense_block(hstR, hstC, &block);
  Eigen::MatrixXd TRblock = hstR.sampletTransform(block);
  Eigen::MatrixXd TTblock = hstC.sampletTransform(TRblock.transpose());
  FMCA::unsymmetric_compressor_impl<H2SampletTree> comp;
  T.tic();
  comp.compress(hstR, hstC, mat_eval, eta, threshold);
  const double tcomp = T.toc("compressor: ");
  auto trips = comp.pattern_triplets();
  std::cout << double(trips.size()) / mpts / npts << std::endl;
  FMCA::SparseMatrix<double> S(mpts, npts);
  S.setFromTriplets(trips.begin(), trips.end());
  std::cout << (TTblock.transpose() - S.full()).norm() / TTblock.norm()
            << std::endl;
  std::cout << std::flush;
#if 0


  {
    Eigen::VectorXd x(npts), y1(npts), y2(npts);
    double err = 0;
    double nrm = 0;
    const double tripSize = sizeof(Eigen::Triplet<double>);
    const double nTrips = symComp.pattern_triplets().size();
    std::cout << "nz(S): " << std::ceil(nTrips / npts) << std::endl;
    std::cout << "memory: " << nTrips * tripSize / 1e9 << "GB\n" << std::flush;
    T.tic();
    for (auto i = 0; i < 100; ++i) {
      unsigned int index = rand() % P.cols();
      x.setZero();
      x(index) = 1;
      y1 = FMCA::matrixColumnGetter(P, hst.indices(), function, index);
      x = hst.sampletTransform(x);
      y2.setZero();
      for (const auto &i : symComp.pattern_triplets()) {
        y2(i.row()) += i.value() * x(i.col());
        if (i.row() != i.col())
          y2(i.col()) += i.value() * x(i.row());
      }
      y2 = hst.inverseSampletTransform(y2);
      err += (y1 - y2).squaredNorm();
      nrm += y1.squaredNorm();
    }
    const double thet = T.toc("matrix vector time: ");
    std::cout << "average matrix vector time " << thet / 100 << "sec."
              << std::endl;
    err = sqrt(err / nrm);
    std::cout << "compression error: " << err << std::endl << std::flush;
  }
  std::cout << std::string(60, '-') << std::endl;
  return 0;
#endif
}
