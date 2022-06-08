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

#include <Eigen/Dense>
#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/Samplets/omp_samplet_compressor.h>
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>

struct expKernel {
  template <typename derived, typename otherDerived>
  FMCA::Scalar operator()(const Eigen::MatrixBase<derived> &x,
                          const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-(x - y).norm());
  }
};

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, expKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main(int argc, char *argv[]) {
  const FMCA::Index dim = atoi(argv[1]);
  const FMCA::Index dtilde = 3;
  const auto function = expKernel();
  const FMCA::Scalar eta = 0.8;
  const FMCA::Index mp_deg = 4;
  const FMCA::Scalar threshold = 0;
  FMCA::Tictoc T;
  for (FMCA::Index npts : {1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6}) {
  //for (FMCA::Index npts : {1e4}) {
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
    const H2SampletTree hst(mom, samp_mom, 0, P);
    T.toc("tree setup: ");
    std::cout << std::flush;
    T.tic();
    FMCA::Matrix data = FMCA::Matrix::Random(P.cols(), 100);
    FMCA::ompSampletTransformer<H2SampletTree> omp_transform(hst);
    T.toc("omp init: ");
    T.tic();
    auto tnew_data = omp_transform.transform(data);
    T.toc("transform: ");
    T.tic();
    auto told_data = hst.sampletTransform(data);
    T.toc("transform old: ");
    std::cout << (tnew_data - told_data).norm() / told_data.norm() << std::endl;
    T.tic();
    auto ttnew_data = omp_transform.inverseTransform(tnew_data);
    T.toc("inverse transform: ");
    std::cout << (ttnew_data - data).norm() / data.norm() << std::endl;
    FMCA::ompSampletCompressor<H2SampletTree> comp;
    T.tic();
    comp.init(hst, 0.8);
    T.toc("omp init: ");
    T.tic();
    comp.compress(hst, mat_eval, threshold);
    T.toc("new compressor: ");
    FMCA::symmetric_compressor_impl<H2SampletTree> symComp;
    T.tic();
    //symComp.compress(hst, mat_eval, eta, threshold);
    T.toc("old compressor: ");
    auto trips1 = comp.pattern_triplets();
    auto trips2 = symComp.pattern_triplets();
    Eigen::SparseMatrix<double> S1(npts, npts);
    Eigen::SparseMatrix<double> S2(npts, npts);

    S1.setFromTriplets(trips1.begin(), trips1.end());
    S2.setFromTriplets(trips2.begin(), trips2.end());

#if 0
    FMCA::symmetric_compressor_impl<H2SampletTree> symComp;
    T.tic();
    symComp.compress(hst, mat_eval, eta, threshold);
    const FMCA::Scalar tcomp = T.toc("symmetric compressor: ");
    std::cout << std::flush;

    {
      Eigen::VectorXd x(npts), y1(npts), y2(npts);
      FMCA::Scalar err = 0;
      FMCA::Scalar nrm = 0;
      const FMCA::Scalar tripSize = sizeof(Eigen::Triplet<FMCA::Scalar>);
      const FMCA::Scalar nTrips = symComp.pattern_triplets().size();
      std::cout << "nz(S): " << std::ceil(nTrips / npts) << std::endl;
      std::cout << "memory: " << nTrips * tripSize / 1e9 << "GB\n"
                << std::flush;
      T.tic();
      for (auto i = 0; i < 100; ++i) {
        FMCA::Index index = rand() % P.cols();
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
      const FMCA::Scalar thet = T.toc("matrix vector time: ");
      std::cout << "average matrix vector time " << thet / 100 << "sec."
                << std::endl;
      err = sqrt(err / nrm);
      std::cout << "compression error: " << err << std::endl << std::flush;
    }
#endif
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}
