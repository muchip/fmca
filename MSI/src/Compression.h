// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Sara Avesani, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//

#ifndef FMCA_MULTIGRID_COMPRESSION_H_
#define FMCA_MULTIGRID_COMPRESSION_H_

#include <Eigen/Sparse>
#include <vector>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"

namespace FMCA {

struct CompressionStats {
  Scalar time_planner = 0.0;      // Time for compression planning
  Scalar time_compressor = 0.0;   // Time for actual compression
  Scalar time_apriori = 0.0;      // Time for a priori compression
  Scalar time_apost = 0.0;        // Time for a posteriori compression
  size_t triplets_apriori = 0;    // Number of a priori triplets per row
  size_t triplets_apost = 0;      // Number of a posteriori triplets per row
  Scalar compression_error = 0.0; // Relative compression error
};

struct CompressionResult {
  Eigen::SparseMatrix<Scalar> matrix;  // Compressed sparse matrix
  CompressionStats stats;              // Compression statistics
};

/**
 * \ingroup Multigrid
 * \brief Compresses covariance matrix using unsymmetric samplet compression
 */
inline CompressionResult UnsymmetricCompressorWithStats(
    const NystromMoments<TotalDegreeInterpolator>& mom_rows,
    const NystromSampletMoments<MonomialInterpolator>& samp_mom_rows,
    const H2SampletTree<ClusterTree>& hst_rows,
    const CovarianceKernel& function, Scalar eta, Scalar threshold_kernel,
    Scalar threshold_aPost, Scalar mpole_deg, Scalar dtilde,
    const Matrix& P_rows, const Matrix& P_cols) {
  
  CompressionResult result;
  CompressionStats& stats = result.stats;
  Tictoc timer;

  // Initialize column structures
  const NystromMoments<TotalDegreeInterpolator> mom_cols(P_cols, mpole_deg);
  const NystromSampletMoments<MonomialInterpolator> samp_mom_cols(P_cols, dtilde - 1);
  const H2SampletTree<ClusterTree> hst_cols(mom_cols, samp_mom_cols, 0, P_cols);

  const int n_pts_rows = P_rows.cols();
  const int n_pts_cols = P_cols.cols();

  // Matrix evaluator setup
  const unsymmetricNystromEvaluator<NystromMoments<TotalDegreeInterpolator>,
                                    CovarianceKernel> mat_eval(mom_rows, mom_cols, function);

  // Compression phases
  timer.tic();
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree<ClusterTree>> K_comp;
  K_comp.init(hst_rows, hst_cols, eta, (n_pts_cols < 100) ? 0 : threshold_kernel);
  stats.time_planner = timer.toc();

  timer.tic();
  K_comp.compress(mat_eval);
  stats.time_compressor = timer.toc();

  // A priori compression
  timer.tic();
  auto triplets_apriori = K_comp.triplets();
  stats.time_apriori = timer.toc();
  stats.triplets_apriori = std::round(triplets_apriori.size() / double(n_pts_rows));

  // A posteriori compression
  auto triplets_final = triplets_apriori;
  if (threshold_aPost != -1) {
    timer.tic();
    triplets_final = K_comp.aposteriori_triplets_fast(threshold_aPost);
    stats.time_apost = timer.toc();
    stats.triplets_apost = std::round(triplets_final.size() / double(n_pts_rows));
  }

  // Error evaluation
  Vector x(n_pts_cols), y1(n_pts_rows), y2(n_pts_rows);
  Scalar err = 0, nrm = 0;
  for (int i = 0; i < 100; ++i) {
    const int index = rand() % n_pts_cols;
    x.setZero();
    x[index] = 1.0;

    const Vector col = function.eval(P_rows, P_cols.col(hst_cols.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst_rows.indices(), hst_rows.block_size()));
    y2.setZero();
    
    x = hst_cols.sampletTransform(x);
    for (const auto& trip : triplets_final)
      y2[trip.row()] += trip.value() * x[trip.col()];
    y2 = hst_rows.inverseSampletTransform(y2);

    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  stats.compression_error = std::sqrt(err / nrm);

  // Final matrix assembly
  Eigen::SparseMatrix<Scalar> sparse_matrix(n_pts_rows, n_pts_cols);
  sparse_matrix.setFromTriplets(triplets_final.begin(), triplets_final.end());
  result.matrix = std::move(sparse_matrix);

  return result;
}

/**
 * \ingroup Multigrid
 * \brief Compresses covariance matrix using symmetric samplet compression
 */
inline CompressionResult SymmetricCompressorWithStats(
    const NystromMoments<TotalDegreeInterpolator>& mom,
    const NystromSampletMoments<MonomialInterpolator>& samp_mom,
    const H2SampletTree<ClusterTree>& hst, const CovarianceKernel& function,
    Scalar eta, Scalar threshold_kernel, Scalar threshold_aPost,
    Scalar mpole_deg, Scalar dtilde, const Matrix& P) {
  
  CompressionResult result;
  CompressionStats& stats = result.stats;
  Tictoc timer;
  const int n_pts = P.cols();

  // Matrix evaluator setup
  const NystromEvaluator<NystromMoments<TotalDegreeInterpolator>,
                        CovarianceKernel> mat_eval(mom, function);

  // Compression phases
  timer.tic();
  FMCA::internal::SampletMatrixCompressor<H2SampletTree<ClusterTree>> K_comp;
  K_comp.init(hst, eta, (n_pts < 100) ? 0 : threshold_kernel);
  stats.time_planner = timer.toc();

  timer.tic();
  K_comp.compress(mat_eval);
  stats.time_compressor = timer.toc();

  // A priori compression
  timer.tic();
  auto triplets_apriori = K_comp.triplets();
  stats.time_apriori = timer.toc();
  stats.triplets_apriori = std::round(triplets_apriori.size() / double(n_pts));

  // A posteriori compression
  auto triplets_final = triplets_apriori;
  if (threshold_aPost != -1) {
    timer.tic();
    triplets_final = K_comp.aposteriori_triplets_fast(threshold_aPost);
    stats.time_apost = timer.toc();
    stats.triplets_apost = std::round(triplets_final.size() / double(n_pts));
  }

  // Error evaluation
  Vector x(n_pts), y1(n_pts), y2(n_pts);
  Scalar err = 0, nrm = 0;
  for (int i = 0; i < 100; ++i) {
    const int index = rand() % n_pts;
    x.setZero();
    x[index] = 1.0;

    const Vector col = function.eval(P.leftCols(n_pts), P.col(hst.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst.indices(), hst.block_size()));
    y2.setZero();
    
    x = hst.sampletTransform(x);
    for (const auto& trip : triplets_final)
      y2[trip.row()] += trip.value() * x[trip.col()];
    y2 = hst.inverseSampletTransform(y2);

    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  stats.compression_error = std::sqrt(err / nrm);

  // Final matrix assembly
  Eigen::SparseMatrix<Scalar> sparse_matrix(n_pts, n_pts);
  sparse_matrix.setFromTriplets(triplets_final.begin(), triplets_final.end());
  result.matrix = std::move(sparse_matrix);

  return result;
}

}  // namespace FMCA

#endif  // FMCA_MULTIGRID_COMPRESSION_H_