// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
// A/B comparison: master SampletMatrixCompressor (Eigen matrices) vs
// SampletMatrixCompressorPool (MemoryPool-backed blocks and scratch).
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"
#include "../FMCA/src/Samplets/samplet_matrix_compressor_pool.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 500000
#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 1.);
  const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  const FMCA::Scalar eta = 0.5;
  const FMCA::Scalar threshold = 100 * FMCA_ZERO_TOLERANCE;

  for (int dtilde = 2; dtilde <= 5; ++dtilde) {
    const FMCA::Index mpole_deg = 2 * (dtilde - 1);
    const Moments mom(P, mpole_deg);
    const MatrixEvaluator mat_eval(mom, function);
    const SampletMoments samp_mom(P, dtilde - 1);
    H2SampletTree hst(mom, samp_mom, 0, P);
    std::cout << "dtilde: " << dtilde << "   mpole_deg: " << mpole_deg
              << std::endl;

    FMCA::internal::SampletMatrixCompressor<H2SampletTree,
                                            FMCA::CompareClusterStrict>
        Sref;
    T.tic();
    Sref.init(hst, eta, threshold);
    T.toc("eigen planner:              ");
    T.tic();
    Sref.compress(mat_eval);
    const FMCA::Scalar t_ref_c = T.toc("eigen compressor:           ");
    T.tic();
    const auto &trips_ref = Sref.triplets();
    const FMCA::Scalar t_ref_t = T.toc("eigen triplets:             ");

    FMCA::internal::SampletMatrixCompressorPool<H2SampletTree,
                                                FMCA::CompareClusterStrict>
        Spool;
    T.tic();
    Spool.init(hst, eta, threshold);
    T.toc("pool  planner:              ");
    T.tic();
    Spool.compress(mat_eval);
    const FMCA::Scalar t_pool = T.toc("pool  compressor+triplets:  ");
    const auto &trips_pool = Spool.triplets();

    std::cout << "nnz (upper): eigen " << trips_ref.size() << "   pool "
              << trips_pool.size() << std::endl;
    Eigen::SparseMatrix<FMCA::Scalar> A(NPTS, NPTS), B(NPTS, NPTS);
    A.setFromTriplets(trips_ref.begin(), trips_ref.end());
    B.setFromTriplets(trips_pool.begin(), trips_pool.end());
    std::cout << "rel. Frobenius error:       " << (A - B).norm() / A.norm()
              << std::endl;
    std::cout << "pool footprint:             " << Spool.pool_footprint() / 1e6
              << " MB" << std::endl;
    std::cout << "speedup (compress+trips):   "
              << (t_ref_c + t_ref_t) / t_pool << "x" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}
