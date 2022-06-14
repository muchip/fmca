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
  const FMCA::Index dtilde = 4;
  const auto function = expKernel();
  const FMCA::Scalar eta = 0.8;
  const FMCA::Index mp_deg = 6;
  const FMCA::Scalar threshold = 1e-5;
  FMCA::Tictoc T;
  // for (FMCA::Index npts : {1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7}) {
  for (FMCA::Index npts : {1e3, 5e3, 1e4, 5e4}) {
    std::cout << "N:                        " << npts << std::endl
              << "dim:                      " << dim << std::endl
              << "eta:                      " << eta << std::endl
              << "multipole degree:         " << mp_deg << std::endl
              << "vanishing moments:        " << dtilde << std::endl
              << "aposteriori threshold:    " << threshold << std::endl;
    T.tic();
    const Eigen::MatrixXd P =
        0.5 * Eigen::MatrixXd::Random(dim, npts).array() + 0.5;
    T.toc("geometry generation:     ");
    const Moments mom(P, mp_deg);
    const MatrixEvaluator mat_eval(mom, function);
    const SampletMoments samp_mom(P, dtilde - 1);
    T.tic();
    const H2SampletTree hst(mom, samp_mom, 0, P);
    T.toc("tree setup:              ");
    std::cout << std::flush;
    FMCA::ompSampletCompressor<H2SampletTree> comp;
    comp.init(hst, 0.8, threshold);
    T.toc("omp initializer:         ");
    comp.compress(hst, mat_eval);
    T.toc("cummulative compressor:  ");
    T.tic();
    const auto &trips = comp.triplets();
    T.toc("generating triplets:     ");
    std::cout << std::flush;

    {
      FMCA::Vector x(npts), y1(npts), y2(npts);
      FMCA::Scalar err = 0;
      FMCA::Scalar nrm = 0;
      const FMCA::Scalar tripSize = sizeof(Eigen::Triplet<FMCA::Scalar>);
      const FMCA::Scalar nTrips = trips.size();
      std::cout << "nz(S):                    " << std::ceil(nTrips / npts)
                << std::endl;
      std::cout << "nnz                       " << nTrips / npts / npts * 100
                << "\%" << std::endl;
      std::cout << "memory:                   " << nTrips * tripSize / 1e9
                << "GB" << std::endl;
      T.tic();
      for (auto i = 0; i < 10; ++i) {
        FMCA::Index index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        y1 = FMCA::matrixColumnGetter(P, hst.indices(), function, index);
        x = hst.sampletTransform(x);
        y2.setZero();
        for (const auto &i : trips) {
          y2(i.row()) += i.value() * x(i.col());
          if (i.row() != i.col())
            y2(i.col()) += i.value() * x(i.row());
        }
        y2 = hst.inverseSampletTransform(y2);
        err += (y1 - y2).squaredNorm();
        nrm += y1.squaredNorm();
      }
      const FMCA::Scalar theta = T.toc("matrix vector time:      ");
      std::cout << "ave. matrix vector time:  " << theta / 100 << "sec."
                << std::endl;
      err = sqrt(err / nrm);
      std::cout << "compression error:        " << err << std::endl
                << std::flush;
    }
#if 0
    {
      FMCA::symmetric_compressor_impl<H2SampletTree> symComp;
      T.tic();
      symComp.compress(hst, mat_eval, eta, threshold);
      const FMCA::Scalar tcomp = T.toc("symmetric compressor:    ");
      const auto &trips2 = symComp.pattern_triplets();
      const FMCA::Scalar nTrips = trips2.size();
      std::cout << "old nz(S):                " << std::ceil(nTrips / npts)
                << std::endl;
      FMCA::Vector x(npts), y1(npts), y2(npts);
      FMCA::Scalar err = 0;
      FMCA::Scalar nrm = 0;
      for (auto i = 0; i < 100; ++i) {
        FMCA::Index index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        y1 = FMCA::matrixColumnGetter(P, hst.indices(), function, index);
        x = hst.sampletTransform(x);
        y2.setZero();
        for (const auto &i : trips) {
          y2(i.row()) += i.value() * x(i.col());
          if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
        }
        y2 = hst.inverseSampletTransform(y2);
        err += (y1 - y2).squaredNorm();
        nrm += y1.squaredNorm();
      }
      err = sqrt(err / nrm);
      std::cout << "compression error old:    " << err << std::endl
                << std::flush;
    }
#endif
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}
