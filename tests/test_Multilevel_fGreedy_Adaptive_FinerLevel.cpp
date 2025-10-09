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
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2

using namespace FMCA;

////////////////////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////////////////////
void plotPoints(const std::string& filename, const Matrix& P,
                const Vector& data) {
  Matrix P3D(3, P.cols());
  P3D.setZero();
  P3D.topRows(2) = P;
  FMCA::IO::plotPointsColor(filename, P3D, data);
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

void runMultigridTest(Scalar nu, Scalar adaptive_threshold = 1e-3,
                      Scalar f_greedy_threshold = 1e-4) {
  std::cout << "======== Running Multigrid Test with nu = " << nu
            << ", adaptive_threshold = " << adaptive_threshold
            << " ========" << std::endl;

  ////////////////////////////// Points
  std::vector<int> gridSizes = {9, 25, 81, 289, 1089, 4225, 16641};
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
  std::vector<Scalar> computational_time_greedy_updates;

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
    // std::cout << "Solver Stats:" << std::endl;
    // std::cout << "- Iterations: " << cg_stats.iterations << std::endl;
    // std::cout << "- Residual: " << cg_stats.residual_error << std::endl;
    // std::cout << "- Solver time: " << cg_stats.solver_time << " s" << std::endl;

    levels.push_back(l + 1);
    N_total.push_back(P_Full[l].cols());
    N_active.push_back(n_pts);
    assembly_time.push_back(compressor_stats.assembly_time);
    cg_time.push_back(cg_stats.solver_time);
    iterationsCG.push_back(cg_stats.iterations);
    anz.push_back(compressor_stats.anz);

    ////////////////////////////// Store solution and solver
    ALPHA.push_back(solution);
    ///////////////////////////////////////////////////////// f greedy for NEXT
    ///level
    if (l + 1 < max_level) {
      std::cout << "\n--- Preparing next level (f-greedy) ---" << std::endl;

      const Matrix& P_candidates = P_Full[max_level - 1];
      int n_candidates = P_candidates.cols();
      Matrix P_next = P_l;  // Initialize with current level's points
      std::set<Index> selected_set;

      // Mark which candidate points are already in P_l
      for (int i = 0; i < P_l.cols(); ++i) {
        for (int j = 0; j < n_candidates; ++j) {
          if ((P_l.col(i) - P_candidates.col(j)).norm() < 1e-10) {
            selected_set.insert(j);
            break;
          }
        }
      }

      std::cout << "Starting with " << P_l.cols()
                << " points from current level" << std::endl;

      // Evaluate current approximation on all candidate points
      Vector target = evalFrankeFunction(P_candidates);
      Vector current_approx = Vector::Zero(n_candidates);

      // Add contributions from all levels (0 to l)
      for (int j = 0; j <= l; ++j) {
        Scalar sigma_j = nu * fill_distances[j];
        CovarianceKernel kernel_j(kernel_type, sigma_j);
        MultipoleFunctionEvaluator evaluator;
        evaluator.init(kernel_j, P_Active[j], P_candidates);
        Vector correction =
            evaluator.evaluate(P_Active[j], P_candidates, ALPHA[j]);
        correction *= std::pow(sigma_j, -DIM);
        current_approx += correction;
      }

      // Compute initial errors
      Vector errors = (target - current_approx).cwiseAbs();
      // Mark already selected points with zero error
      for (Index idx : selected_set) {
        errors(idx) = 0;
      }
      Scalar error = errors.maxCoeff();
      Index counter_new_points = P_l.cols();

      std::cout << "Initial max error: " << error << std::endl;
      Scalar threshold = f_greedy_threshold * pow(0.5, l+1);
      std::cout << "f-greedy threshold: "
                << threshold << std::endl;

      ///////////////////////////////////////////// Greedy loop: add points until error threshold is met
      Tictoc time_greedy_update;
      time_greedy_update.tic();

      while (error > threshold &&
             counter_new_points < n_candidates) {
        Index max_idx;
        Scalar max_error = errors.maxCoeff(&max_idx);

        if (std::isnan(max_error) || std::isinf(max_error)) {
          std::cout
              << "WARNING: Invalid error detected, stopping greedy selection"
              << std::endl;
          break;
        }

        if (selected_set.count(max_idx) > 0) {
          errors(max_idx) = 0;
          continue;
        }

        // Add point to selected set
        selected_set.insert(max_idx);
        counter_new_points++;

        // Resize P_next and add the new point
        Matrix P_next_new(DIM, counter_new_points);
        P_next_new.leftCols(counter_new_points - 1) = P_next;
        P_next_new.col(counter_new_points - 1) = P_candidates.col(max_idx);
        P_next = P_next_new;

        // Compute residual at P_next
        Vector residual_next = evalFrankeFunction(P_next);
        for (int j = 0; j < l; ++j) {
          Scalar sigma_j = nu * fill_distances[j];
          CovarianceKernel kernel_j(kernel_type, sigma_j);
          MultipoleFunctionEvaluator evaluator;
          evaluator.init(kernel_j, P_Active[j], P_next);
          Vector correction = evaluator.evaluate(P_Active[j], P_next, ALPHA[j]);
          correction *= std::pow(sigma_j, -DIM);
          residual_next -= correction;
        }

        // Solve with SAME kernel as level l (sigma_l)
        MultilevelSampletKernelSolver<> solver_next;
        solver_next.init(kernel_l, P_next, dtilde, eta, threshold, ridgep);
        solver_next.compress(P_next);

        Vector solution_next = solver_next.solveIterative(
            residual_next, preconditioner, cg_threshold);
        solution_next /= std::pow(sigma_l, -DIM);

        // Evaluate on all candidate points
        MultipoleFunctionEvaluator evaluator_next;
        evaluator_next.init(kernel_l, P_next, P_candidates);
        Vector new_level_approx =
            evaluator_next.evaluate(P_next, P_candidates, solution_next);
        new_level_approx *= std::pow(sigma_l, -DIM);

        // Update current approximation
        current_approx = Vector::Zero(n_candidates);
        for (int j = 0; j < l; ++j) {
          Scalar sigma_j = nu * fill_distances[j];
          CovarianceKernel kernel_j(kernel_type, sigma_j);
          MultipoleFunctionEvaluator evaluator;
          evaluator.init(kernel_j, P_Active[j], P_candidates);
          Vector correction =
              evaluator.evaluate(P_Active[j], P_candidates, ALPHA[j]);
          correction *= std::pow(sigma_j, -DIM);
          current_approx += correction;
        }
        current_approx += new_level_approx;

        // Recompute errors
        errors = (target - current_approx).cwiseAbs();
        for (Index idx : selected_set) {
          errors(idx) = 0;
        }

        error = errors.maxCoeff();

        // if (counter_new_points % 100 == 0) {
        //   std::cout << "Selected " << counter_new_points
        //             << " points, max error: " << error << std::endl;
        // }
      }
      Scalar time_greedy = time_greedy_update.toc("Time for f-greedy update: ");
      computational_time_greedy_updates.push_back(time_greedy);

      std::cout << "\nf-greedy selection complete:" << std::endl;
      std::cout << "- Total points: " << counter_new_points << " / "
                << n_candidates << " ("
                << (100.0 * counter_new_points / n_candidates) << "%)"
                << std::endl;
      std::cout << "- New points added: " << (counter_new_points - P_l.cols())
                << std::endl;
      std::cout << "- Final max error: " << error << std::endl;

      P_Active.push_back(P_next);

      std::stringstream plot_filename;
      plot_filename << "active_points_fgreedy_level_" << (l + 2) << ".vtk";
      plotPoints(plot_filename.str(), P_next, Vector::Ones(P_next.cols()));
      std::cout << "Saved active points to: " << plot_filename.str()
                << std::endl;
    }
  }  // End of adaptive loop

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

    // std::cout << "Level " << (l + 1) << ": L2 = " << l2_err
    //           << ", Linf = " << linf_err << std::endl;
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
  std::cout << "======== Computational time updates ========" << std::endl;
  // print it as a python array
  std::cout << "[";
  for (size_t i = 0; i < computational_time_greedy_updates.size(); ++i) {
    std::cout << computational_time_greedy_updates[i] << ", ";
  }
  std::cout << "]" << std::endl;
}

////////////////////////////// MAIN
int main() {
  Scalar nu = 1.0;
  Scalar adaptive_threshold = 1e-1;
  Scalar f_greedy_threshold = 1e-2;

  runMultigridTest(nu, adaptive_threshold, f_greedy_threshold);

  return 0;
}