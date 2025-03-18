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

#ifndef FMCA_MULTIGRID_SOLVER_H_
#define FMCA_MULTIGRID_SOLVER_H_

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <stdexcept>

#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"

namespace FMCA {

struct SolverStats {
  Scalar solver_time = 0.0;     // Total solver execution time
  int iterations = 0;           // Number of iterations used
  Scalar residual_error = 0.0;  // Final residual error
};

struct SolverResult {
  Vector solution;      // Computed solution vector
  SolverStats stats;    // Solver statistics
};

/**
 * \ingroup Multigrid
 * \brief Solves linear system using iterative methods with statistics tracking
 */
inline SolverResult solveSystemWithStats(
    const Eigen::SparseMatrix<Scalar, Eigen::RowMajor, long long int>& A_comp,
    const Vector& rhs, const std::string& solverName,
    Scalar threshold_CG = 1e-6) {
  SolverResult result;
  Tictoc timer;
  
  // Create selfadjoint view for symmetric solvers
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor, long long int> A_sym = A_comp.selfadjointView<Eigen::Upper>();

  if (solverName == "ConjugateGradient") {
    timer.tic();
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar, Eigen::RowMajor, long long int>, Eigen::Lower|Eigen::Upper,
                             Eigen::IdentityPreconditioner> solver;
    solver.setTolerance(threshold_CG);
    solver.compute(A_sym);
    
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("CG: Matrix decomposition failed");
    
    result.solution = solver.solve(rhs);
    result.stats.iterations = solver.iterations();
    result.stats.solver_time = timer.toc();
  }
  else if (solverName == "ConjugateGradientwithPreconditioner") {
    timer.tic();
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar, Eigen::RowMajor, long long int>, Eigen::Lower|Eigen::Upper> solver;
    solver.setTolerance(threshold_CG);
    solver.compute(A_sym);
    
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("CG with Preconditioner: Decomposition failed");
    
    result.solution = solver.solve(rhs);
    result.stats.iterations = solver.iterations();
    result.stats.solver_time = timer.toc();
  }
  else {
    throw std::invalid_argument("Invalid solver name. Options are: "
                                "'ConjugateGradient' or "
                                "'ConjugateGradientwithPreconditioner'");
  }

  // Calculate final residual
  result.stats.residual_error = (A_sym * result.solution - rhs).norm();
  
  return result;
}

}  // namespace FMCA

#endif  // FMCA_MULTIGRID_SOLVER_H_