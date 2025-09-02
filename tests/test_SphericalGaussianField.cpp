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
#define FMCA_MATERNNU
#include <iostream>
#include <random>
//
#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/NormalDistribution.h"
#include "../FMCA/src/util/Tictoc.h"

using Cholesky = Eigen::SimplicialLLT<Eigen::SparseMatrix<FMCA::Scalar>,
                                      Eigen::Upper, Eigen::MetisOrdering<int>>;

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
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

int main() {
  FMCA::Tictoc T;
  //////////////////////////////////////////////////////////////////////////////
  FMCA::CovarianceKernel function("MaternNu", 0.5, 1., 0.25);
  function.setDistanceType("GEODESIC");
  //////////////////////////////////////////////////////////////////////////////
  FMCA::Matrix P = FibonacciLattice(200000);
  FMCA::Index npts = P.cols();
  const FMCA::Scalar threshold = 5e-4;
  const FMCA::Scalar eta = .1;
  const FMCA::Scalar dtilde = 3;
  const FMCA::Index mpole_deg = 4;
  const Moments mom(P, mpole_deg);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  FMCA::NormalDistribution nd(0, 1, 0);
  FMCA::Matrix PP(3, npts);
  FMCA::Matrix PN(3, npts);
  {
    FMCA::Index npos = 0;
    FMCA::Index nneg = 0;
    for (FMCA::Index i = 0; i < P.cols(); ++i)
      if (P(2, i) > 0)
        PP.col(npos++) = P.col(i);
      else
        PN.col(nneg++) = P.col(i);
    PP.conservativeResize(3, npos);
    PN.conservativeResize(3, nneg);
  }
  // P = PP;
  // npts = P.cols();
  //////////////////////////////////////////////////////////////////////////////
  std::cout << "dtilde:                       " << dtilde << std::endl;
  std::cout << "mpole_deg:                    " << mpole_deg << std::endl;
  std::cout << "eta:                          " << eta << std::endl;
  H2SampletTree hst(mom, samp_mom, 0, P);
  {
    FMCA::Vector data = function.eval(P, P.col(0));
    FMCA::IO::plotPointsColor("data.vtk", P, data);
  }
#if 0
  std::vector<FMCA::Matrix> bbvec;
  for (auto &&it : hst) {
    if (!it.nSons() && it.block_size()) {
      for (auto j = 0; j < it.block_size(); ++j)
        assert(FMCA::internal::inBoundingBox(it, P.col(it.indices()[j])) &&
               "point outside leaf bounding box");
      bbvec.push_back(it.bb());
    }
  }
  FMCA::IO::plotBoxes("boxes.vtk", bbvec);
  FMCA::IO::plotPoints("points.vtk", P);
#endif
  T.tic();
  FMCA::internal::SampletMatrixCompressor<H2SampletTree,
                                          FMCA::CompareSphericalCluster>
      Scomp;
  Scomp.init(hst, eta, 1e-7);
  T.toc("planner:                     ");
  T.tic();
  Scomp.compress(mat_eval);
  T.toc("compressor:                  ");
  T.tic();
  const auto &ap_trips = Scomp.triplets();
  std::cout << "anz (a-priori):               "
            << std::round(ap_trips.size() / FMCA::Scalar(npts)) << std::endl;
  T.toc("triplets:                    ");

  T.tic();
  const auto &trips = Scomp.aposteriori_triplets_fast(threshold);
  std::cout << "anz (a-posteriori):           "
            << std::round(trips.size() / FMCA::Scalar(npts)) << std::endl;
  T.toc("triplets:                    ");
  Eigen::SparseMatrix<FMCA::Scalar> S(npts, npts);
  S.setFromTriplets(trips.begin(), trips.end());

  FMCA::Vector x(npts), y1(npts), y2(npts);
  FMCA::Scalar err = 0;
  FMCA::Scalar nrm = 0;
  for (auto i = 0; i < 10; ++i) {
    FMCA::Index index = rand() % P.cols();
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
  T.tic();
  Cholesky llt_;
  llt_.compute(S);
  if (llt_.info() != Eigen::Success) {
    std::cout << "Factorization failed!" << std::endl;
  }
  std::cout << "sinfo: " << llt_.info() << std::endl;
  std::cout << "nz(L): "
            << llt_.matrixL().nestedExpression().nonZeros() / FMCA::Scalar(npts)
            << std::endl;
  T.toc("factorization done: ");
  FMCA::Vector sample;
  for (FMCA::Index i = 0; i < 1; ++i) {
    FMCA::Vector randn = nd.randN(npts, 1);
    FMCA::Vector TLrandn =
        llt_.permutationPinv() * (llt_.matrixL() * randn).eval();
    FMCA::Vector Lrandn = hst.inverseSampletTransform(TLrandn);
    sample = hst.toNaturalOrder(Lrandn);
  }
  FMCA::IO::plotPointsColor("sample.vtk", P, sample);
  std::cout << std::string(60, '-') << std::endl;
  return 0;
}
