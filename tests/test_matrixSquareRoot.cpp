// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
//
#include <FMCA/Kernel>
#include <FMCA/Samplets>
//
#include <FMCA/src/util/FibonacciLattice.h>
#include <FMCA/src/util/IO.h>
#include <FMCA/src/util/NormalDistribution.h>
#include <FMCA/src/util/RandomTreeAccessor.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/matrixSquareRoot.h>

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator =
    FMCA::NystromEvaluator<Moments,
                           FMCA::PDKernel<FMCA::Metric::GeodesicSphere>>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::SphereClusterTree>;

FMCA::Matrix FibonacciLattice(const FMCA::Index N) {
  FMCA::Matrix retval(3, N);
  const FMCA::Scalar golden_angle = FMCA_PI * (3.0 - std::sqrt(5.0));
  for (FMCA::Index i = 0; i < N; ++i) {
    const FMCA::Scalar z = 1.0 - (2.0 * i + 1.0) / N;
    const FMCA::Scalar radius = std::sqrt(1.0 - z * z);
    const FMCA::Scalar phi = golden_angle * i;
    const FMCA::Scalar x = radius * std::cos(phi);
    const FMCA::Scalar y = radius * std::sin(phi);
    retval.col(i) << x, y, z;
  }
  return retval;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  //////////////////////////////////////////////////////////////////////////////
  FMCA::PDKernel<FMCA::Metric::GeodesicSphere> function("Exponential", .25);
  const FMCA::Index npts = 10000;
  FMCA::Matrix P = FibonacciLattice(npts);
  const FMCA::Scalar threshold = 1e-4;
  const FMCA::Scalar eta = .1;
  const FMCA::Scalar dtilde = 3;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const Moments mom(P, mpole_deg);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);

  H2SampletTree hst(mom, samp_mom, 0, P);
  FMCA::SampletMatrixCompressor<H2SampletTree, FMCA::CompareSphericalCluster>
      Scomp;
  Scomp.init(hst, eta, 1e-7);
  Scomp.compress(mat_eval);
  const auto &ap_trips = Scomp.triplets();
  const auto &trips = Scomp.aposteriori_triplets_fast(threshold);
  Eigen::SparseMatrix<FMCA::Scalar> S(npts, npts);
  S.setFromTriplets(trips.begin(), trips.end());
  FMCA::Vector x(S.cols()), y1(S.rows()), y2(S.rows());
  FMCA::Scalar err = 0;
  FMCA::Scalar nrm = 0;
  for (auto i = 0; i < 10; ++i) {
    FMCA::Index index = rand() % npts;
    x.setZero();
    x(index) = 1;
    FMCA::Vector col = function.eval(P, P.col(hst.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst.indices(), hst.block_size()));
    x = hst.sampletTransform(x);
    y2.setZero();
    y2 = S.selfadjointView<Eigen::Upper>() * x;
    y2 = hst.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  err = sqrt(err / nrm);
  std::cout << "compression error:            " << err << std::endl
            << std::flush;
  err = 0;
  nrm = 0;
  for (auto i = 0; i < 10; ++i) {
    x.setRandom();
    FMCA::Vector Sx = S.selfadjointView<Eigen::Upper>() * x;
    FMCA::Vector Rx =
        FMCA::matrixSquareRoot(S.selfadjointView<Eigen::Upper>(), x, 20);
    FMCA::Vector RRx =
        FMCA::matrixSquareRoot(S.selfadjointView<Eigen::Upper>(), Rx, 20);
    err += (Sx - RRx).squaredNorm();
    nrm += Sx.squaredNorm();
  }
  err = sqrt(err / nrm);
  std::cout << "matsqrt error:                " << err << std::endl
            << std::flush;
#if 0
  FMCA::NormalDistribution nd(0, 1, 0);
  FMCA::Vector Nrand = nd.randN(npts, 1);
  FMCA::Vector Rx =
      FMCA::matrixSquareRoot(S.selfadjointView<Eigen::Upper>(), Nrand, 20);
  FMCA::Vector sample = hst.toNaturalOrder(hst.inverseSampletTransform(Rx));
  FMCA::IO::plotPointsColor("surf.vtk", P, sample);
#endif
  return 0;
}
