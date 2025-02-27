#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
// ##############################
#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Plotter.h"
#include "../FMCA/src/util/Plotter3D.h"
#include "../FMCA/src/util/Tictoc.h"

using namespace FMCA;

struct CompressionStats {
  double time_planner = 0.0;         // from T.tic(), K_comp.init(...)
  double time_compressor = 0.0;      // from K_comp.compress(...)
  double time_apriori = 0.0;         // from triplets extraction
  std::size_t triplets_apriori = 0;  // number of a-priori triplets
};

// 3) A container for the final matrix + stats
struct CompressionResult {
  Eigen::SparseMatrix<Scalar> matrix;  // The resulting sparse matrix
  CompressionStats stats;              // The times + stats from compression
};

// 4) A struct to store solver stats (time + #iterations + solution)
struct SolverStats {
  double solver_time = 0.0;
  int iterations = 0;
};

struct SolverResult {
  Eigen::VectorXd solution;
  SolverStats stats;
};

CompressionResult UnsymmetricCompressorWithStats(
    const NystromMoments<TotalDegreeInterpolator>& mom_rows,
    const NystromSampletMoments<MonomialInterpolator>& samp_mom_rows,
    const H2SampletTree<ClusterTree>& hst_rows,
    const CovarianceKernel& function, const Scalar& eta,
    const Scalar& threshold_kernel, const Scalar& threshold_aPost,
    const Scalar& mpole_deg, const Scalar& dtilde, const Matrix& P_rows,
    const Matrix& P_cols) {
  CompressionResult result;  // will return .matrix and .stats
  CompressionStats& stats = result.stats;

  // Build the columns side
  const NystromMoments<TotalDegreeInterpolator> mom_cols(P_cols, mpole_deg);
  const NystromSampletMoments<MonomialInterpolator> samp_mom_cols(P_cols,
                                                                  dtilde - 1);
  const H2SampletTree<ClusterTree> hst_cols(mom_cols, samp_mom_cols, 0, P_cols);

  int n_pts_rows = P_rows.cols();
  int n_pts_cols = P_cols.cols();

  // Evaluator
  const unsymmetricNystromEvaluator<NystromMoments<TotalDegreeInterpolator>,
                                    FMCA::CovarianceKernel>
      mat_eval(mom_rows, mom_cols, function);

  // Tictoc local
  Tictoc T;
  T.tic();
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree<ClusterTree>>
      K_comp;
  // Decide threshold based on col-size
  if (n_pts_cols < 100) {
    K_comp.init(hst_rows, hst_cols, eta, 0);
  } else {
    K_comp.init(hst_rows, hst_cols, eta, threshold_kernel);
  }
  stats.time_planner = T.toc();  // planner time

  T.tic();
  K_comp.compress(mat_eval);
  stats.time_compressor = T.toc();  // compressor time

  T.tic();
  const auto& triplets_aPriori = K_comp.triplets();
  stats.time_apriori = T.toc();  // time to get a-priori triplets
  stats.triplets_apriori = triplets_aPriori.size();

  // Print some info
  std::cout << "\nanz (a-priori) unsymmetric: "
            << std::round(triplets_aPriori.size() / double(n_pts_rows))
            << std::endl;
  std::cout << "time (a-priori) unsymmetric: " << stats.time_apriori
            << " sec.\n";

  // If user wants a-posteriori threshold
  std::vector<Eigen::Triplet<Scalar>> triplets_aPost;
  if (threshold_aPost != -1) {
    // not adding to the times the user specifically wanted
    // sum of planner+compression+time(apriori)
    // but we do it here if needed
    T.tic();
    triplets_aPost = K_comp.aposteriori_triplets(threshold_aPost);
    double time_aPost = T.toc();
    std::cout << "anz (a-posteriori) unsymmetric: "
              << std::round(triplets_aPost.size() / double(n_pts_rows))
              << std::endl;
    std::cout << "time (a-posteriori) unsymmetric: " << time_aPost << " sec.\n";
  }

  // Quick compression error check (optional, as in your code)
  Vector x(n_pts_cols), y1(n_pts_rows), y2(n_pts_rows);
  Scalar err = 0, nrm = 0;
  for (int i = 0; i < 100; ++i) {
    int index = rand() % n_pts_cols;
    x.setZero();
    x(index) = 1.0;
    Vector col = function.eval(P_rows, P_cols.col(hst_cols.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst_rows.indices(),
                                             hst_rows.block_size()));
    x = hst_cols.sampletTransform(x);
    y2.setZero();

    if (threshold_aPost != -1) {
      for (const auto& trip : triplets_aPost) {
        y2(trip.row()) += trip.value() * x(trip.col());
      }
    } else {
      for (const auto& trip : triplets_aPriori) {
        y2(trip.row()) += trip.value() * x(trip.col());
      }
    }
    y2 = hst_rows.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  std::cout << "Compression error unsymmetric:  " << std::sqrt(err / nrm)
            << std::endl;

  // Build the final matrix
  Eigen::SparseMatrix<Scalar> Kcomp_Sparse(n_pts_rows, n_pts_cols);
  if (threshold_aPost != -1) {
    Kcomp_Sparse.setFromTriplets(triplets_aPost.begin(), triplets_aPost.end());
  } else {
    Kcomp_Sparse.setFromTriplets(triplets_aPriori.begin(),
                                 triplets_aPriori.end());
  }
  Kcomp_Sparse.makeCompressed();
  result.matrix = Kcomp_Sparse;

  return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
CompressionResult SymmetricCompressorWithStats(
    const NystromMoments<TotalDegreeInterpolator>& mom,
    const NystromSampletMoments<MonomialInterpolator>& samp_mom,
    const H2SampletTree<ClusterTree>& hst, const CovarianceKernel& function,
    const Scalar& eta, const Scalar& threshold_kernel,
    const Scalar& threshold_aPost, const Scalar& mpole_deg,
    const Scalar& dtilde, const Matrix& P) {
  CompressionResult result;  // .matrix and .stats
  CompressionStats& stats = result.stats;

  int n_pts = P.cols();
  const NystromEvaluator<NystromMoments<TotalDegreeInterpolator>,
                         CovarianceKernel>
      mat_eval(mom, function);

  Tictoc T;
  T.tic();
  FMCA::internal::SampletMatrixCompressor<H2SampletTree<ClusterTree>> K_comp;
  K_comp.init(hst, eta, threshold_kernel);
  stats.time_planner = T.toc();  // planner

  T.tic();
  K_comp.compress(mat_eval);
  stats.time_compressor = T.toc();  // compressor

  T.tic();
  const auto& triplets_aPriori = K_comp.triplets();
  stats.time_apriori = T.toc();
  stats.triplets_apriori = triplets_aPriori.size();

  std::cout << "\nanz (a-priori) symmetric: "
            << std::round(triplets_aPriori.size() / double(n_pts)) << std::endl;
  std::cout << "time (a-priori) symmetric: " << stats.time_apriori << " sec.\n";

  // A-posteriori if needed
  std::vector<Eigen::Triplet<Scalar>> triplets_aPost;
  if (threshold_aPost != -1) {
    T.tic();
    triplets_aPost = K_comp.aposteriori_triplets(threshold_aPost);
    double time_aPost = T.toc();
    std::cout << "anz (a-posteriori) symmetric: "
              << std::round(triplets_aPost.size() / double(n_pts)) << std::endl;
    std::cout << "time (a-posteriori) symmetric: " << time_aPost << " sec.\n";
  }

  // Quick check of compression error
  Vector x(n_pts), y1(n_pts), y2(n_pts);
  Scalar err = 0, nrm = 0;
  for (int i = 0; i < 100; ++i) {
    Index index = rand() % n_pts;
    x.setZero();
    x(index) = 1;
    Vector col = function.eval(P.leftCols(n_pts), P.col(hst.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst.indices(), hst.block_size()));
    x = hst.sampletTransform(x);
    y2.setZero();

    // For symmetric we must also fill i.col() + i.row().
    if (threshold_aPost != -1) {
      for (const auto& t : triplets_aPost) {
        y2(t.row()) += t.value() * x(t.col());
        if (t.row() != t.col()) {
          y2(t.col()) += t.value() * x(t.row());
        }
      }
    } else {
      for (const auto& t : triplets_aPriori) {
        y2(t.row()) += t.value() * x(t.col());
        if (t.row() != t.col()) {
          y2(t.col()) += t.value() * x(t.row());
        }
      }
    }
    y2 = hst.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  std::cout << "Compression error symmetric:   " << std::sqrt(err / nrm)
            << std::endl;

  // Build final matrix
  Eigen::SparseMatrix<Scalar> Kcomp_Sparse(n_pts, n_pts);
  if (threshold_aPost != -1) {
    Kcomp_Sparse.setFromTriplets(triplets_aPost.begin(), triplets_aPost.end());
  } else {
    Kcomp_Sparse.setFromTriplets(triplets_aPriori.begin(),
                                 triplets_aPriori.end());
  }
  Kcomp_Sparse.makeCompressed();
  result.matrix = Kcomp_Sparse;

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////
Vector EvaluateWithStats(
    const NystromMoments<TotalDegreeInterpolator>& mom_eval,
    const NystromSampletMoments<MonomialInterpolator>& samp_mom_eval,
    const H2SampletTree<ClusterTree>& hst_eval, const std::string& kernel_type,
    const std::vector<Matrix>& P_Matrices, const Matrix& Peval,
    const std::vector<Vector>& ALPHA, const Vector& fill_distances,
    const Scalar& max_level, const Scalar& nu, const Scalar& eta,
    const Scalar& threshold_kernel, const Scalar& threshold_aPost,
    const Scalar& mpole_deg,
    const Scalar& dtilde, const Vector& exact_sol,
    const H2SampletTree<ClusterTree>& hst, const std::string& base_filename) {
  Vector solution = Vector::Zero(Peval.cols());
  for (Index i = 0; i < max_level; ++i) {
    std::cout << "-------- Evaluation Level " << i + 1 << " --------"
              << std::endl;
    Scalar sigma = nu * fill_distances[i];
    const CovarianceKernel function(kernel_type, sigma);

    // UnsymmetricCompressor to evaluate
    CompressionResult cr = UnsymmetricCompressorWithStats(
        mom_eval, samp_mom_eval, hst_eval, function, eta, threshold_kernel,
        threshold_aPost, mpole_deg, dtilde, Peval, P_Matrices[i]);

    // multiply K_eval * alpha
    Vector partial = cr.matrix * ALPHA[i];
    solution += partial;

    // get solution in natural basis just to measure final error
    Vector solution_natural_basis = hst.inverseSampletTransform(solution);
    solution_natural_basis = hst.toNaturalOrder(solution_natural_basis);

    // Optional: Plot
    if (!base_filename.empty()) {
      if (Peval.rows() == 2) {
        std::ostringstream oss;
        oss << base_filename << "_level_" << i << ".vtk";
        Plotter2D plotter;
        plotter.plotFunction2D(oss.str(), Peval, solution_natural_basis);
      } else if (Peval.rows() == 3) {
        std::ostringstream oss;
        oss << base_filename << "_level_" << i << ".vtk";
        Plotter3D plotter;
        plotter.plotFunction(oss.str(), Peval, solution_natural_basis);
      }
    }

    // Evaluate error
    Vector diff_abs(solution_natural_basis.rows());
    for (Index k = 0; k < solution_natural_basis.rows(); ++k) {
      diff_abs[k] = std::abs(solution_natural_basis[k] - exact_sol[k]);
    }
    std::cout << "Error L2   = "
              << (solution_natural_basis - exact_sol).norm() / exact_sol.norm()
              << std::endl;
    std::cout << "Error L_inf= " << diff_abs.maxCoeff() << std::endl
              << std::endl;
  }
  return solution;
}

//////////////////////////////////////////////////////////////////////////////////////

SolverResult solveSystemWithStats(const Eigen::SparseMatrix<double>& A_comp,
  const Vector& rhs,
  const std::string& solverName,
  double threshold_CG = 1e-6) {
SolverResult result;
SolverStats& stats = result.stats;

// If using CG, we want to keep track of #iterations
Eigen::SparseMatrix<double> A_sym = A_comp.selfadjointView<Eigen::Upper>();

Tictoc T;
if (solverName == "SimplicialLLT") {
T.tic();
Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper,
Eigen::MetisOrdering<int>>
solver;
solver.compute(A_comp);
if (solver.info() != Eigen::Success) {
throw std::runtime_error("SimplicialLLT: Decomposition failed");
}
result.solution = solver.solve(rhs);
stats.solver_time = T.toc();
stats.iterations = 0;  // direct solver, no iteration
} else if (solverName == "SimplicialLDLT") {
T.tic();
Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
Eigen::MetisOrdering<int>>
solver;
solver.compute(A_comp);
if (solver.info() != Eigen::Success) {
throw std::runtime_error("SimplicialLDLT: Decomposition failed");
}
result.solution = solver.solve(rhs);
stats.solver_time = T.toc();
stats.iterations = 0;  // direct solver
} else if (solverName == "ConjugateGradient") {
T.tic();
Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
Eigen::Lower | Eigen::Upper,
Eigen::IdentityPreconditioner>
solver;
solver.setTolerance(threshold_CG);
solver.compute(A_sym);
if (solver.info() != Eigen::Success) {
throw std::runtime_error("CG: Decomposition failed");
}
result.solution = solver.solve(rhs);
stats.solver_time = T.toc();
stats.iterations = solver.iterations();
} else if (solverName == "ConjugateGradientwithPreconditioner") {
T.tic();
Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
Eigen::Lower | Eigen::Upper>
solver;
solver.setTolerance(threshold_CG);
solver.compute(A_sym);
if (solver.info() != Eigen::Success) {
throw std::runtime_error("CG with Preconditioner: Decomposition failed");
}
result.solution = solver.solve(rhs);
stats.solver_time = T.toc();
stats.iterations = solver.iterations();
} else {
throw std::invalid_argument("Unknown solver name: " + solverName);
}

// Print
std::cout << "Solver: " << solverName << std::endl;
std::cout << "Solver time: " << stats.solver_time << std::endl;
if (stats.iterations > 0) {
std::cout << "#iterations (CG): " << stats.iterations << std::endl;
}
double residual_err = (A_sym * result.solution - rhs).norm();
std::cout << "Residual error: " << residual_err << std::endl << std::endl;

return result;
}