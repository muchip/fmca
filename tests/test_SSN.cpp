// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
// #define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/SSN.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 100000
#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

FMCA::Matrix rhs(const H2SampletTree &hst, const FMCA::SparseMatrix &K) {
  srand(0);
  const FMCA::Index npts = K.cols();
  FMCA::Vector data(npts);
  FMCA::Vector x(npts);
  std::cout << "right hand side:              "
            << "sparse single-scale coeffs" << std::endl;
  x.setZero();
  for (FMCA::Index i = 0; i < 10; ++i) {
    FMCA::Index rdm = rand() % npts;
    while (x(rdm)) rdm = rand() % npts;
    x(rdm) = 1;
  }
  data = npts * hst.inverseSampletTransform(K.selfadjointView<Eigen::Upper>() *
                                            hst.sampletTransform(x));
  data /= 2 * data.cwiseAbs().maxCoeff();
  return data;
}

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("Matern32", 0.25 / sqrt(DIM));
  const FMCA::Matrix P = 0.5 * FMCA::Matrix::Random(DIM, NPTS).array();
  const FMCA::Scalar threshold = 1e-7;
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 5;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const Moments mom(P, mpole_deg);
  const MatrixEvaluator mat_eval(mom, function);
  std::cout << "dtilde:                       " << dtilde << std::endl;
  std::cout << "mpole_deg:                    " << mpole_deg << std::endl;
  std::cout << "eta:                          " << eta << std::endl;
  const SampletMoments samp_mom(P, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.tic();
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Scomp;
  Scomp.init(hst, eta, 100 * FMCA_ZERO_TOLERANCE);
  T.toc("planner:                     ");
  T.tic();
  Scomp.compress(mat_eval);
  T.toc("compressor:                  ");
  T.tic();
  const auto &ap_trips = Scomp.triplets();
  std::cout << "anz (a-priori):               "
            << std::round(ap_trips.size() / FMCA::Scalar(NPTS)) << std::endl;
  T.toc("triplets:                    ");

  T.tic();
  const auto &trips = Scomp.aposteriori_triplets_fast(threshold);
  std::cout << "anz (a-posteriori):           "
            << std::round(trips.size() / FMCA::Scalar(NPTS)) << std::endl;

  T.toc("triplets:                    ");
  {
    FMCA::Vector x(NPTS), y1(NPTS), y2(NPTS);
    FMCA::Scalar err = 0;
    FMCA::Scalar nrm = 0;
    for (auto i = 0; i < 10; ++i) {
      FMCA::Index index = rand() % P.cols();
      x.setZero();
      x(index) = 1;
      FMCA::Vector col = function.eval(P, P.col(hst.indices()[index]));
      y1 =
          col(Eigen::Map<const FMCA::iVector>(hst.indices(), hst.block_size()));
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
    std::cout << "compression error:            " << err << std::endl
              << std::flush;
  }
  Eigen::SparseMatrix<FMCA::Scalar> S(NPTS, NPTS);
  S.setFromTriplets(trips.begin(), trips.end());
  Eigen::SparseMatrix<FMCA::Scalar> Ssym = S.selfadjointView<Eigen::Upper>();
  Ssym *= 1. / FMCA::Scalar(NPTS);
  // const FMCA::Matrix data = rhs(hst, Ssym);
  FMCA::Vector data(NPTS);
  for (FMCA::Index i = 0; i < NPTS; ++i)
    data(i) = std::cos(10 * FMCA_PI * P.col(i).norm()) *
              std::exp(-4 * P.col(i).norm());
  data = hst.toClusterOrder(data);
  FMCA::Matrix P3(3, P.cols());
  P3.topRows(2) = P;
  P3.bottomRows(1) = hst.toNaturalOrder(data).transpose();
  FMCA::IO::plotPointsColor("data.vtk", P3, hst.toNaturalOrder(data));
  FMCA::Matrix Tdata =
      1. / std::sqrt(FMCA::Scalar(NPTS)) * hst.sampletTransform(data);
  FMCA::Vector w(NPTS);
  FMCA::Vector x0(NPTS);
  x0.setZero();
  w.setOnes();
  w *= 1. / std::sqrt(FMCA::Scalar(NPTS));
  FMCA::ActiveSetManager asmgr;
  // x0(0) = 1e4;
  FMCA::Scalar ramp = 1 << 15;

  x0 = FMCA::SSN(Ssym, Tdata, 0.001 * w, x0, asmgr, 200, 1e-8);
  while (ramp > 1) {
    x0 = TRSSN(Ssym, Tdata, 2 * ramp * 1e-10 * w, x0, asmgr, 1e4, 0.01, .5,
               400, 1e-5);
    ramp *= 0.5;
  }

#if 0
  for (FMCA::Index i = 0; i < 10; ++i) {
    x0 = TRSSN(Ssym, Tdata, w, x0, asmgr, 5e-2, 0.05, .05, 200, 1e-6);
    w *= 0.5;
    fac *= 0.5;
  }
#endif
  x0 *= 1. / std::sqrt(FMCA::Scalar(NPTS));
  Tdata = hst.inverseSampletTransform(NPTS * Ssym * x0);
  Tdata = hst.toNaturalOrder(Tdata);
  P3.bottomRows(1) = Tdata.transpose();
  FMCA::IO::plotPointsColor("rec.vtk", P3, Tdata);
  FMCA::Vector err = hst.toNaturalOrder(data) - Tdata;
  std::cout << "error: " << err.norm() / data.norm() << std::endl;
  P3.bottomRows(1) = err.transpose();
  FMCA::IO::plotPointsColor("err.vtk", P3, err);

  return 0;
}