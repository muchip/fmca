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
#include "read_files_txt.h"

#define DIM 3

using namespace FMCA;

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
Scalar Function(Scalar x, Scalar y, Scalar z) {
  Scalar c1_x = -0.02, c1_y = 0.12, c1_z = 0.008;
  Scalar c2_x = 0.03, c2_y = 0.05, c2_z = 0.03;
  Scalar c3_x = -0.07, c3_y = 0.18, c3_z = -0.05;
  Scalar c4_x = -0.06, c4_y = 0.038, c4_z = 0.046;
  Scalar A1 = 10.0, A2 = 10.0, A3 = 10.0, A4 = 10.0;
  Scalar sigma1 = 0.02, sigma2 = 0.02, sigma3 = 0.01, sigma4 = 0.01;

  // Calcolo delle distanze al quadrato dai centri
  Scalar d1_sq = (x - c1_x) * (x - c1_x) + (y - c1_y) * (y - c1_y) +
                 (z - c1_z) * (z - c1_z);
  Scalar d2_sq = (x - c2_x) * (x - c2_x) + (y - c2_y) * (y - c2_y) +
                 (z - c2_z) * (z - c2_z);
  Scalar d3_sq = (x - c3_x) * (x - c3_x) + (y - c3_y) * (y - c3_y) +
                 (z - c3_z) * (z - c3_z);
  Scalar d4_sq = (x - c4_x) * (x - c4_x) + (y - c4_y) * (y - c4_y) +
                 (z - c4_z) * (z - c4_z);

  // Calcolo delle 4 gaussiane
  Scalar g1 = A1 * exp(-d1_sq / (2.0 * sigma1 * sigma1));
  Scalar g2 = A2 * exp(-d2_sq / (2.0 * sigma2 * sigma2));
  Scalar g3 = A3 * exp(-d3_sq / (2.0 * sigma3 * sigma3));
  Scalar g4 = A4 * exp(-d4_sq / (2.0 * sigma4 * sigma4));
  return g1 + g2 + g3 + g4;
}

Vector evalFunction(const Matrix& Points) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = Function(Points(0, i), Points(1, i), Points(2, i));
  }
  return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////
void plotPoints(const std::string& filename, const Matrix& P,
                const Vector& data) {
  FMCA::IO::plotPointsColor(filename, P, data);
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

void runMultigridTest(Scalar nu, Scalar f_greedy_threshold = 1e-4) {
  ////////////////////////////// Points
  std::vector<Matrix> P_Full;
  Matrix P1, P2, P3, P4, Peval;
  readTXT("bunny_level0.txt", P1, 3);
  readTXT("bunny_level1.txt", P2, 3);
  readTXT("bunny_level2.txt", P3, 3);
  readTXT("bunny_level3.txt", P4, 3);
  readTXT("bunny_level4.txt", Peval, 3);
  P_Full.push_back(P1);
  P_Full.push_back(P2);
  P_Full.push_back(P3);
  P_Full.push_back(P4);
  int max_level = P_Full.size();

  std::stringstream plot_filename;
  plot_filename << "function_bunny.vtk";
  plotPoints(plot_filename.str(), Peval, evalFunction(Peval));
  std::cout << "Saved solution to: " << plot_filename.str() << std::endl;

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

  ////////////////////////////// Precompute fill distances of the underlying
  /// grids
  std::vector<Scalar> fill_distances;
  std::cout << "\nPrecomputing fill distances..." << std::endl;
  for (int l = 0; l < max_level; ++l) {
    SampletKernelSolver<> temp_solver;
    temp_solver.init(P_Full[l], dtilde, eta, threshold, ridgep);
    Scalar h = temp_solver.fill_distance();
    fill_distances.push_back(h);
    std::cout << "Level " << (l + 1) << ": h = " << h << std::endl;
  }

  ////////////////////////////// Storage
  std::vector<Matrix> P_Active;  // Active points at each level
  std::vector<Vector> ALPHA;     // Solution coefficients
  std::vector<SampletKernelSolver<>> SOLVERS;

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
    Vector residual = evalFunction(P_l);

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
    SampletKernelSolver<> solver;
    solver.init(P_l, dtilde, eta, threshold, ridgep);
    Tictoc compression_timer;
    compression_timer.tic();
    solver.compress(P_l, kernel_l);
    Scalar compression_time = compression_timer.toc();
    solver.compressionError(P_l, kernel_l);
    std::cout << "\nCompression Stats:" << std::endl;
    std::cout << "- Compression time: " << compression_time << " s"
              << std::endl;
    ////////////////////////////// Solver
    Tictoc solver_timer;
    solver_timer.tic();
    Vector solution =
        solver.solveIteratively(residual, preconditioner, cg_threshold);
    Scalar solver_time = solver_timer.toc();
    solution /= std::pow(sigma_l, -DIM);
    levels.push_back(l + 1);
    N_total.push_back(P_Full[l].cols());
    N_active.push_back(n_pts);
    assembly_time.push_back(compression_time);
    cg_time.push_back(solver_time);
    iterationsCG.push_back(solver.solver_iterations());
    anz.push_back(solver.anz());

    ////////////////////////////// Store solution and solver
    ALPHA.push_back(solution);

    ///////////////////////////////////////////////////////// f greedy for NEXT
    ///level
    if (l + 1 < max_level) {
      std::cout << "\n--- Preparing next level (f-greedy) ---" << std::endl;

      const Matrix& P_candidates = P_Full[l + 1];
      int n_candidates = P_candidates.cols();
      Matrix P_next = P_l;  // Initialize with current level's points
      std::set<Index> selected_set;

      // rescale the kernel for this level
      Scalar sigma_next = nu * fill_distances[l + 1];
      CovarianceKernel kernel_next(kernel_type, sigma_next);

      std::cout << "Next level kernel: sigma = " << sigma_next << std::endl;

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

      Vector target = evalFunction(P_candidates);
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
      Scalar greedy_threshold = f_greedy_threshold * pow(0.5, l + 1);
      std::cout << "f-greedy threshold: " << greedy_threshold << std::endl;

      ///////////////////////////////////////////// Greedy loop
      Tictoc time_greedy_update;
      time_greedy_update.tic();

      while (error > greedy_threshold && counter_new_points < n_candidates) {
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

        Vector residual_next = evalFunction(P_next);
        for (int j = 0; j <= l; ++j) {  // ← j <= l (includi livello corrente)
          Scalar sigma_j = nu * fill_distances[j];
          CovarianceKernel kernel_j(kernel_type, sigma_j);
          MultipoleFunctionEvaluator evaluator;
          evaluator.init(kernel_j, P_Active[j], P_next);
          Vector correction = evaluator.evaluate(P_Active[j], P_next, ALPHA[j]);
          correction *= std::pow(sigma_j, -DIM);
          residual_next -= correction;
        }

        SampletKernelSolver<> solver_next;
        solver_next.init(P_next, dtilde, eta, threshold, ridgep);
        solver_next.compress(P_next, kernel_next);

        Vector solution_next = solver_next.solveIteratively(
            residual_next, preconditioner, cg_threshold);
        solution_next /= std::pow(sigma_next, -DIM);

        MultipoleFunctionEvaluator evaluator_next;
        evaluator_next.init(kernel_next, P_next, P_candidates);
        Vector new_level_approx =
            evaluator_next.evaluate(P_next, P_candidates, solution_next);
        new_level_approx *= std::pow(sigma_next, -DIM);

        // Update current approximation
        current_approx = Vector::Zero(n_candidates);
        for (int j = 0; j <= l; ++j) {  // ← j <= l
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

        if (counter_new_points % 100 == 0) {
          std::cout << "Selected " << counter_new_points
                    << " points, max error: " << error << std::endl;
        }
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
  }

  ////////////////////////////// Evaluation
  std::cout << "\nEvaluating solution on " << Peval.cols() << " points..."
            << std::endl;
  Vector exact_sol = evalFunction(Peval);
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

    std::stringstream plot_filename;
    plot_filename << "solution_fgreedy_level_" << (l + 1) << ".vtk";
    plotPoints(plot_filename.str(), Peval, final_res);
    std::cout << "Saved solution to: " << plot_filename.str() << std::endl;

    Scalar l2_err = (final_res - exact_sol).norm() / exact_sol.norm();
    Scalar linf_err = (final_res - exact_sol).cwiseAbs().maxCoeff();
    l2_errors.push_back(l2_err);
    linf_errors.push_back(linf_err);
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
  std::cout << "[";
  for (size_t i = 0; i < computational_time_greedy_updates.size(); ++i) {
    std::cout << computational_time_greedy_updates[i];
    if (i < computational_time_greedy_updates.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
}

////////////////////////////// MAIN
int main() {
  Scalar nu = 0.5;
  Scalar f_greedy_threshold = 1e-1;

  runMultigridTest(nu, f_greedy_threshold);
  return 0;
}