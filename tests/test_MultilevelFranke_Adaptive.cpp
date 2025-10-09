#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/KernelInterpolation"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/IO.h"

#define DIM 2

using namespace FMCA;

// Franke Function
Scalar FrankeFunction(Scalar x, Scalar y) {
  return (0.5 * std::tanh(2000 * (x + y - 1)));
}

Vector evalFrankeFunction(const Matrix& Points) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = FrankeFunction(Points(0, i), Points(1, i));
  }
  return f;
}

// Function to generate uniform grid points on unit square
Matrix generateUniformGrid(int n) {
  int gridSize = std::sqrt(n);
  Matrix P(DIM, gridSize * gridSize);

  Scalar h = 1.0 / (gridSize - 1);
  int idx = 0;

  for (int i = 0; i < gridSize; ++i) {
    for (int j = 0; j < gridSize; ++j) {
      P(0, idx) = i * h;
      P(1, idx) = j * h;
      idx++;
    }
  }

  return P;
}

void plotPoints(const std::string& filename, const Matrix& P,
                const Vector& data) {
  Matrix P3D(3, P.cols());
  P3D.setZero();
  P3D.topRows(2) = P;
  FMCA::IO::plotPointsColor(filename, P3D, data);
}

//////////////////////////////////////////////////////////////////////////////////////////
// Check if point is in bounding box
template <typename Derived>
bool isPointInBoundingBox(const Matrix& point, const Derived& cluster) {
  const auto& bb_min = cluster.bb().col(0);
  const auto& bb_max = cluster.bb().col(1);
  for (Index d = 0; d < DIM; ++d) {
    if (point(d, 0) < bb_min(d) || point(d, 0) > bb_max(d)) {
      return false;
    }
  }
  return true;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Get active leaves from tree based on residual
template <typename Derived>
std::vector<const Derived*> getActiveLeaves(
    const SampletTreeBase<Derived>& tree, const Vector& residual,
    Scalar threshold) {
  // Transform to samplet coefficients
  Vector r_sorted = tree.toClusterOrder(residual);
  Vector r_samplet = tree.sampletTransform(r_sorted);

  // Adaptive tree search
  Scalar norm2 = r_samplet.squaredNorm();
  std::vector<const Derived*> active_clusters =
      adaptiveTreeSearch(tree, r_samplet, threshold * norm2);

  // Extract only leaves
  std::vector<const Derived*> active_leaves;
  for (const auto* cluster : active_clusters) {
    if (cluster != nullptr && cluster->nSons() == 0) {
      active_leaves.push_back(cluster);
    }
  }

  return active_leaves;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Extract points from P_next that are inside active leaves
template <typename Derived>
Matrix extractPointsInActiveRegions(
    const Matrix& P_next, const std::vector<const Derived*>& active_leaves) {
  std::set<Index> active_indices;

  for (Index i = 0; i < P_next.cols(); ++i) {
    Matrix point = P_next.col(i);
    for (const auto* leaf : active_leaves) {
      if (leaf != nullptr && isPointInBoundingBox(point, *leaf)) {
        active_indices.insert(i);
        break;
      }
    }
  }

  Matrix P_active(DIM, active_indices.size());
  Index idx = 0;
  for (Index i : active_indices) {
    P_active.col(idx++) = P_next.col(i);
  }

  return P_active;
}

void runMultigridTest(Scalar nu, Scalar adaptive_threshold = 1e-3) {
  std::cout << "======== Running Multigrid Test with nu = " << nu
            << ", adaptive_threshold = " << adaptive_threshold
            << " ========" << std::endl;

  ////////////////////////////// Points
  std::vector<int> gridSizes = {9, 25, 81, 289, 1089, 4225, 16641, 66049, 262145};
  int max_level = gridSizes.size();
  std::vector<Matrix> P_Full;
  for (int size : gridSizes) {
    P_Full.push_back(generateUniformGrid(size));
  }
  Matrix Peval = generateUniformGrid(40000);

  ////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 4;
  const Scalar threshold = 1e-8;
  const std::string kernel_type = "matern32";
  const Scalar ridgep = 0;
  const bool preconditioner = true;
  Scalar cg_threshold = 1e-6;

  std::cout << "Parameters:" << std::endl;
  std::cout << "- eta:              " << eta << std::endl;
  std::cout << "- dtilde:           " << dtilde << std::endl;
  std::cout << "- threshold:        " << threshold << std::endl;
  std::cout << "- kernel_type:      " << kernel_type << std::endl;
  std::cout << "- nu:               " << nu << std::endl;
  std::cout << "- adaptive_threshold: " << adaptive_threshold << std::endl;

  ////////////////////////////// Precompute fill distances of the underlying
  /// grids
  std::vector<Scalar> fill_distances;
  std::cout << "\nPrecomputing fill distances..." << std::endl;
  for (int l = 0; l < max_level; ++l) {
    MultilevelSampletKernelSolver<> temp_solver;
    CovarianceKernel temp_kernel(kernel_type, 1);
    temp_solver.init(temp_kernel, P_Full[l], dtilde, eta, threshold, ridgep);
    Scalar h = temp_solver.fill_distance();
    fill_distances.push_back(h);
    std::cout << "Level " << (l + 1) << ": h = " << h << std::endl;
  }

  ////////////////////////////// Storage
  std::vector<Matrix> P_Active;  // Active points at each level
  std::vector<Vector> ALPHA;     // Solution coefficients
  std::vector<MultilevelSampletKernelSolver<>> SOLVERS;

  // Initialize level 0 with all points
  P_Active.push_back(P_Full[0]);

  ////////////////////////////// Summary results
  std::vector<int> levels;
  std::vector<int> N_total;
  std::vector<int> N_active;
  std::vector<Scalar> assembly_time;
  std::vector<Scalar> cg_time;
  std::vector<int> iterationsCG;
  std::vector<Scalar> anz;

  ////////////////////////////// Adaptive Loop
  for (int l = 0; l < max_level; ++l) {
    std::cout << std::endl;
    std::cout << "-------- LEVEL " << (l + 1) << " --------" << std::endl;

    Matrix P_l = P_Active[l];
    int n_pts = P_l.cols();
    std::cout << "Number of points: " << n_pts << " / " << P_Full[l].cols()
              << " (" << (100.0 * n_pts / P_Full[l].cols()) << "%)"
              << std::endl;
    std::cout << "Fill distance: " << fill_distances[l] << std::endl;

    ////////////////////////////// Compute residual
    Vector residual = evalFrankeFunction(P_l);

    for (int j = 0; j < l; ++j) {
      Scalar sigma_j = nu * fill_distances[j];
      CovarianceKernel kernel_j(kernel_type, sigma_j);
      MultipoleFunctionEvaluator evaluator;
      evaluator.init(kernel_j, P_Active[j], P_l);
      Vector correction = evaluator.evaluate(P_Active[j], P_l, ALPHA[j]);
      correction *= std::pow(sigma_j, -DIM);
      residual -= correction;
    }

    std::cout << "Residual norm: " << residual.norm() << std::endl;

    ////////////////////////////// Compress the diagonal block
    Scalar sigma_l = nu * fill_distances[l];
    CovarianceKernel kernel_l(kernel_type, sigma_l);

    MultilevelSampletKernelSolver<> solver;
    solver.init(kernel_l, P_l, dtilde, eta, threshold, ridgep);
    solver.compress(P_l);
    solver.compressionError(P_l);

    const auto& compressor_stats = solver.getCompressionStats();
    std::cout << "\nCompression Stats:" << std::endl;
    std::cout << "- Compression time: " << compressor_stats.time_compressor
              << " s" << std::endl;
    std::cout << "- Compression error: " << compressor_stats.compression_error
              << std::endl;

    ////////////////////////////// Solver
    Vector solution =
        solver.solveIterative(residual, preconditioner, cg_threshold);
    solution /= std::pow(sigma_l, -DIM);

    const auto& cg_stats = solver.getSolverStats();
    std::cout << "Solver Stats:" << std::endl;
    std::cout << "- Iterations: " << cg_stats.iterations << std::endl;
    std::cout << "- Residual: " << cg_stats.residual_error << std::endl;
    std::cout << "- Solver time: " << cg_stats.solver_time << " s" << std::endl;

    levels.push_back(l + 1);
    N_total.push_back(P_Full[l].cols());
    N_active.push_back(n_pts);
    assembly_time.push_back(compressor_stats.assembly_time);
    cg_time.push_back(cg_stats.solver_time);
    iterationsCG.push_back(cg_stats.iterations);
    anz.push_back(compressor_stats.anz);

    ////////////////////////////// Store solution and solver
    ALPHA.push_back(solution);

    ////////////////////////////// Adaptive refinement for NEXT level
    if (l + 1 < max_level) {
      std::cout << "\n--- Preparing next level ---" << std::endl;
      const auto& tree = solver.getSampletTree();
      auto active_leaves = getActiveLeaves(tree, residual, adaptive_threshold);

      std::cout << "Active leaves: " << active_leaves.size() << std::endl;

      // Extract points from next grid in active regions
      Matrix P_next =
          extractPointsInActiveRegions(P_Full[l + 1], active_leaves);

      std::cout << "Selected " << P_next.cols() << " / " << P_Full[l + 1].cols()
                << " points (" << (100.0 * P_next.cols() / P_Full[l + 1].cols())
                << "%) for next level" << std::endl;

      P_Active.push_back(P_next);

      ////////////////////////////// Plot active points
      std::stringstream plot_filename;
      plot_filename << "active_points_level_" << (l + 1) << ".vtk";
      plotPoints(plot_filename.str(), P_next, Vector::Ones(P_next.cols()));
      std::cout << "Saved active points to: " << plot_filename.str()
                << std::endl;
    }
  }

  ////////////////////////////// Evaluation
  std::cout << "\nEvaluating solution on " << Peval.cols() << " points..."
            << std::endl;
  Vector exact_sol = evalFrankeFunction(Peval);
  Vector final_res = Vector::Zero(Peval.cols());
  std::vector<Scalar> l2_errors;
  std::vector<Scalar> linf_errors;

  for (int l = 0; l < max_level; ++l) {
    Scalar sigma = nu * fill_distances[l];
    CovarianceKernel kernel(kernel_type, sigma);
    MultipoleFunctionEvaluator evaluator;
    evaluator.init(kernel, P_Active[l], Peval);
    Vector eval = evaluator.evaluate(P_Active[l], Peval, ALPHA[l]);
    eval *= std::pow(sigma, -DIM);
    final_res += eval;

    Scalar l2_err = (final_res - exact_sol).norm() / exact_sol.norm();
    Scalar linf_err = (final_res - exact_sol).cwiseAbs().maxCoeff();
    l2_errors.push_back(l2_err);
    linf_errors.push_back(linf_err);

    std::cout << "Level " << (l + 1) << ": L2 = " << l2_err
              << ", Linf = " << linf_err << std::endl;
  }

  ////////////////////////////// Results Summary
  std::cout << "\n======== Results Summary ========" << std::endl;
  std::cout << std::left << std::setw(10) << "Level" << std::setw(10)
            << "N_total" << std::setw(10) << "N_active" << std::setw(10)
            << "Ratio%" << std::setw(15) << "AssemblyTime" << std::setw(15)
            << "CGTime" << std::setw(10) << "IterCG" << std::setw(10) << "ANZ"
            << std::setw(15) << "L2 Error" << std::setw(15) << "Linf Error"
            << std::endl;

  for (size_t i = 0; i < levels.size(); ++i) {
    Scalar ratio = 100.0 * N_active[i] / N_total[i];
    std::cout << std::left << std::setw(10) << levels[i] << std::setw(10)
              << N_total[i] << std::setw(10) << N_active[i] << std::setw(10)
              << std::fixed << std::setprecision(1) << ratio << std::setw(15)
              << std::fixed << std::setprecision(6) << assembly_time[i]
              << std::setw(15) << std::fixed << std::setprecision(6)
              << cg_time[i] << std::setw(10) << iterationsCG[i] << std::setw(10)
              << static_cast<int>(anz[i]) << std::setw(15) << std::scientific
              << std::setprecision(6) << l2_errors[i] << std::setw(15)
              << std::scientific << std::setprecision(6) << linf_errors[i]
              << std::endl;
  }
}

////////////////////////////// MAIN
int main() {
  Scalar nu = 1.0;
  Scalar adaptive_threshold = 1e-1;

  runMultigridTest(nu, adaptive_threshold);

  return 0;
}
