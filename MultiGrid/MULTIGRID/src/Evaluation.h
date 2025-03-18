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

#ifndef FMCA_MULTIGRID_EVALUATION_H_
#define FMCA_MULTIGRID_EVALUATION_H_

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "Compression.h"

namespace FMCA {

struct EvaluationStats {
  Scalar eval_time = 0.0;         // Total evaluation time
  Scalar compression_time = 0.0;  // Time spent in compression
  Scalar assembly_time = 0.0;     // Time spent in assembly
  Scalar matvec_time = 0.0;       // Time spent in matrix-vector products
};

struct EvaluationResult {
  Vector solution;          // Evaluated solution
  Scalar l2_error = 0.0;    // L2 error if exact solution provided
  Scalar linf_error = 0.0;  // L-infinity error if exact solution provided
  EvaluationStats stats;    // Timing statistics
};

/**
 * \ingroup Multigrid
 * \brief Evaluates the multigrid solution with detailed statistics
 */
inline EvaluationResult EvaluateWithStats(
    const NystromMoments<TotalDegreeInterpolator>& mom_eval,
    const NystromSampletMoments<MonomialInterpolator>& samp_mom_eval,
    const H2SampletTree<ClusterTree>& hst_eval, const std::string& kernel_type,
    const std::vector<Matrix>& P_Matrices, const Matrix& Peval,
    const std::vector<Vector>& ALPHA, const Vector& fill_distances,
    const Scalar& max_level, const Scalar& nu, const Scalar& eta,
    const Scalar& threshold_kernel, const Scalar& threshold_aPost,
    const Scalar& mpole_deg, const Scalar& dtilde, const Vector& exact_sol,
    const std::string& base_filename, const Scalar& dim) {
  
  EvaluationResult result;
  result.solution = Vector::Zero(Peval.cols());
  
  Tictoc timer;
  timer.tic();
  
  for (Index i = 0; i < max_level; ++i) {
    std::cout << "-------- Evaluation Level " << i + 1 << " --------\n";
    const Scalar sigma = nu * fill_distances[i];
    const CovarianceKernel function(kernel_type, sigma);
    
    // Compression phase
    Tictoc compression_timer;
    compression_timer.tic();
    CompressionResult cr = UnsymmetricCompressorWithStats(
        mom_eval, samp_mom_eval, hst_eval, function, eta,
        threshold_kernel, threshold_aPost, mpole_deg, dtilde, Peval, P_Matrices[i]);
    cr.matrix *= std::pow(sigma, -dim);
    result.stats.compression_time += compression_timer.toc();
    
    // Matrix-vector product
    Tictoc matvec_timer;
    matvec_timer.tic();
    result.solution += cr.matrix * ALPHA[i];
    result.stats.matvec_time += matvec_timer.toc();
    
    // Assembly and error calculation
    Tictoc assembly_timer;
    assembly_timer.tic();
    Vector solution_natural_basis = hst_eval.inverseSampletTransform(result.solution);
    solution_natural_basis = hst_eval.toNaturalOrder(solution_natural_basis);
    result.stats.assembly_time += assembly_timer.toc();
    
    // Error computation
    const Vector diff_abs = (solution_natural_basis - exact_sol).cwiseAbs();
    result.l2_error = (solution_natural_basis - exact_sol).norm() / exact_sol.norm();
    result.linf_error = diff_abs.maxCoeff();
    
    std::cout << "Error L2   = " << result.l2_error << "\n"
              << "Error L_inf= " << result.linf_error << "\n\n";
  }
  
  result.stats.eval_time = timer.toc();
  return result;
}

}  // namespace FMCA

#endif  // FMCA_MULTIGRID_EVALUATION_H_