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

void runMultigridTest(Scalar nu, Scalar f_greedy_threshold = 1e-4,
                      int batch_size = 100) {
  ////////////////////////////// Points
  int max_steps = 4;
  Matrix P0;
  readTXT("bunny_level0.txt", P0, 3);
  Matrix P_finest; 
  readTXT("bunny_level4.txt", P_finest, 3);
  Matrix Peval;
  readTXT("bunny_level5.txt", Peval, 3);
  int n_candidates = P_finest.cols();
  Vector target = evalFunction(P_finest);
  ////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 4;
  const Scalar threshold = 1e-8;
  const std::string kernel_type = "matern32";
  const Scalar ridgep = 0;
  const bool preconditioner = true;
  Scalar cg_threshold = 1e-6;
  Scalar tau = 3. / 2 + DIM / 2;

  std::cout << "Parameters:" << std::endl;
  std::cout << "- eta:              " << eta << std::endl;
  std::cout << "- dtilde:           " << dtilde << std::endl;
  std::cout << "- threshold:        " << threshold << std::endl;
  std::cout << "- kernel_type:      " << kernel_type << std::endl;
  std::cout << "- nu:               " << nu << std::endl;
  std::cout << "- max_steps:        " << max_steps << std::endl;

  ////////////////////////////// Storage
  std::vector<Matrix> P_Active;  // Active points at each level
  std::vector<Vector> ALPHA;     // Solution coefficients

  // Initialize level 0 with all points
  P_Active.push_back(P0);
  Scalar n0 = P0.cols();

  ////////////////////////////// Summary results
  std::vector<int> levels;
  std::vector<int> N_total;
  std::vector<int> N_active;
  std::vector<Scalar> assembly_time;
  std::vector<Scalar> cg_time;
  std::vector<int> iterationsCG;
  std::vector<Scalar> anz;
  std::vector<Scalar> computational_time_greedy_updates;

  //////////////////////////////
  ////////////////////////////// Multigrid loop
  for (int l = 0; l < max_steps; ++l) {
    std::cout << std::endl;
    std::cout << "-------- LEVEL " << (l + 1) << " --------" << std::endl;
    // start with the active points at level l
    Matrix P_l = P_Active[l];
    int n_pts = P_l.cols();
    std::cout << "Number of points: " << n_pts << " / " << P_finest.cols()
              << " (" << (100.0 * n_pts / P_finest.cols()) << "%)" << std::endl;
    ////////////////////////////// Compute residual
    Vector residual = evalFunction(P_l);
    for (int j = 0; j < l; ++j) {
      Scalar sigma_j = nu * pow(n0, -1. / DIM) * pow(2, -j);
      CovarianceKernel kernel_j(kernel_type, sigma_j);
      MultipoleFunctionEvaluator evaluator;
      evaluator.init(kernel_j, P_Active[j], P_l);
      Vector correction = evaluator.evaluate(P_Active[j], P_l, ALPHA[j]);
      correction *= std::pow(sigma_j, -DIM);
      residual -= correction;
    }
    ////////////////////////////// Compress the diagonal block + stats
    Scalar sigma_l = nu * pow(n0, -1. / DIM) * pow(2, -l);
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
    ////////////////////////////// Solver + stats
    Tictoc solver_timer;
    solver_timer.tic();
    Vector solution =
        solver.solveIteratively(residual, preconditioner, cg_threshold);
    Scalar solver_time = solver_timer.toc();
    solution /= std::pow(sigma_l, -DIM);
    levels.push_back(l + 1);
    N_total.push_back(P_finest.cols());
    N_active.push_back(n_pts);
    assembly_time.push_back(compression_time);
    cg_time.push_back(solver_time);
    iterationsCG.push_back(solver.solver_iterations());
    anz.push_back(solver.anz());
    ////////////////////////////// Store solution
    ALPHA.push_back(solution);
    residual.resize(0);  // I do not need it anymore

    //////////////////////////////
    ////////////////////////////// f greedy for NEXT level
    if (l + 1 < max_steps) {
      std::cout << "\n--- Preparing next level (f-greedy) ---" << std::endl;
      int initial_size = P_l.cols();
      int current_size = initial_size;  // start with the current level points
      int points_to_add = (std::pow(2, DIM) - 1) * initial_size;
      int batched_points_to_add =
          ((points_to_add + batch_size - 1) / batch_size) * batch_size;
      int target_size =
          initial_size +
          batched_points_to_add;  // to be sure it maches the batch size

      Matrix P_next(DIM, target_size);
      P_next.leftCols(initial_size) = P_l;

      // Marca i punti già selezionati
      std::vector<bool> is_selected(n_candidates, false);
      for (int i = 0; i < initial_size; ++i) {
        for (int j = 0; j < n_candidates; ++j) {
          if ((P_next.col(i) - P_finest.col(j)).norm() < 1e-10) {
            is_selected[j] = true;
            break;
          }
        }
      }

      // next level kernel
      Scalar sigma_next = nu * pow(n0, -1. / DIM) * pow(2, -(l + 1));
      CovarianceKernel kernel_next(kernel_type, sigma_next);
      std::cout << "Starting with " << initial_size
                << " points from current level" << std::endl;
      std::cout << "Target size: " << target_size << " points" << std::endl;
      std::cout << "Next level kernel: sigma = " << sigma_next << std::endl;

      ///////////////////////////////////////////// Greedy loop
      Tictoc time_greedy_update;
      time_greedy_update.tic();
      int iteration = 0;
      Scalar error_threshold = f_greedy_threshold * pow(pow(0.5, tau), l + 1);
      std::cout << "Error threshold: " << error_threshold << std::endl;

      while (current_size < target_size && current_size < n_candidates) {
        Vector solution_next;
        Index max_idx = -1;
        Scalar max_error = 0.0;

        // Scope for solver and residual computation
        {
          Vector residual_next = evalFunction(P_next.leftCols(current_size));
          for (int j = 0; j <= l; ++j) {
            Scalar sigma_j = nu * pow(n0, -1. / DIM) * pow(2, -j);
            CovarianceKernel kernel_j(kernel_type, sigma_j);
            MultipoleFunctionEvaluator evaluator;
            evaluator.init(kernel_j, P_Active[j],
                           P_next.leftCols(current_size));
            Vector correction = evaluator.evaluate(
                P_Active[j], P_next.leftCols(current_size), ALPHA[j]);
            correction *= std::pow(sigma_j, -DIM);
            residual_next -= correction;
          }

          // solve
          SampletKernelSolver<> solver_next;
          solver_next.init(P_next.leftCols(current_size), dtilde, eta,
                           threshold, ridgep);
          solver_next.compress(P_next.leftCols(current_size), kernel_next);
          solution_next = solver_next.solveIteratively(
              residual_next, preconditioner, cg_threshold);
          solution_next /= std::pow(sigma_next, -DIM);
        }  // solver_next and residual_next destroyed here

        // Scope for evaluation and error computation
        Vector errors;
        {
          Vector full_approx = Vector::Zero(n_candidates);
          for (int j = 0; j <= l; ++j) {
            Scalar sigma_j = nu * pow(n0, -1. / DIM) * pow(2, -j);
            CovarianceKernel kernel_j(kernel_type, sigma_j);
            MultipoleFunctionEvaluator evaluator;
            evaluator.init(kernel_j, P_Active[j], P_finest);
            Vector correction =
                evaluator.evaluate(P_Active[j], P_finest, ALPHA[j]);
            correction *= std::pow(sigma_j, -DIM);
            full_approx += correction;
          }

          MultipoleFunctionEvaluator evaluator_next;
          evaluator_next.init(kernel_next, P_next.leftCols(current_size),
                              P_finest);
          Vector next_level_contrib = evaluator_next.evaluate(
              P_next.leftCols(current_size), P_finest, solution_next);
          next_level_contrib *= std::pow(sigma_next, -DIM);
          full_approx += next_level_contrib;

          errors = (target - full_approx).cwiseAbs();
        }  // full_approx, errors, and evaluators destroyed here

        std::vector<Index> candidates_above_threshold;
        for (Index k = 0; k < n_candidates; ++k) {
          if (!is_selected[k] && errors(k) > error_threshold) {
            candidates_above_threshold.push_back(k);
          }
        }
        // std::cout << "Number of candidates above threshold: "
        //           << candidates_above_threshold.size() << std::endl;
        if (candidates_above_threshold.size() == 0) {
          std::cout << "No points above threshold" << std::endl;
          break;
        }

        int points_to_select =
            std::min(batch_size, (int)candidates_above_threshold.size());
        // select batched_points_to_add points uniformely distributed among them
        for (int i = 0; i < points_to_select; ++i) {
          Scalar max_min_dist = -1.0;
          Index best_idx = -1;

          // Per ogni candidato, trova la distanza minima dai punti già
          // selezionati
          for (Index k : candidates_above_threshold) {
            if (is_selected[k]) continue;

            Scalar min_dist = std::numeric_limits<Scalar>::max();
            for (int j = 0; j < current_size; ++j) {
              Scalar dist = (P_finest.col(k) - P_next.col(j)).norm();
              min_dist = std::min(min_dist, dist);
            }

            if (min_dist > max_min_dist) {
              max_min_dist = min_dist;
              best_idx = k;
            }
          }

          if (best_idx != -1) {
            P_next.col(current_size) = P_finest.col(best_idx);
            is_selected[best_idx] = true;
            current_size++;
          }
        }
      }

      std::cout << "\nf-greedy selection complete:" << std::endl;
      std::cout << "- Total points: " << current_size << " / " << n_candidates
                << " (" << (100.0 * current_size / n_candidates) << "%)"
                << std::endl;
      std::cout << "- New points added: " << (current_size - initial_size)
                << std::endl;

      // Salva solo le colonne effettivamente riempite
      if (current_size < target_size) {
        P_Active.push_back(P_next.leftCols(current_size));
      } else {
        P_Active.push_back(P_next);
      }

      std::stringstream plot_filename;
      plot_filename << "active_points_fgreedy_batches_level_" << (l + 2)
                    << ".vtk";
      plotPoints(plot_filename.str(), P_next.leftCols(current_size),
                 Vector::Ones(current_size));
      std::cout << "Saved active points to: " << plot_filename.str()
                << std::endl;

      Scalar time_greedy = time_greedy_update.toc("Time for f-greedy update: ");
      computational_time_greedy_updates.push_back(time_greedy);
    }
  }
  //////////////////////////////
  ////////////////////////////// Evaluation
  std::cout << "\nEvaluating solution on " << Peval.cols() << " points..."
            << std::endl;
  Vector exact_sol = evalFunction(Peval);
  Vector final_res = Vector::Zero(Peval.cols());
  std::vector<Scalar> l2_errors;
  std::vector<Scalar> linf_errors;

  for (int l = 0; l < max_steps; ++l) {
    Scalar sigma = nu * pow(n0, -1. / DIM) * pow(2, -l);
    CovarianceKernel kernel(kernel_type, sigma);
    MultipoleFunctionEvaluator evaluator;
    evaluator.init(kernel, P_Active[l], Peval);
    Vector eval = evaluator.evaluate(P_Active[l], Peval, ALPHA[l]);
    eval *= std::pow(sigma, -DIM);
    final_res += eval;

    std::stringstream plot_filename;
    plot_filename << "solution_fgreedy_batches_level_" << (l + 1) << ".vtk";
    plotPoints(plot_filename.str(), Peval, final_res);
    std::cout << "Saved solution to: " << plot_filename.str() << std::endl;

    Scalar l2_err = (final_res - exact_sol).norm() / exact_sol.norm();
    Scalar linf_err = (final_res - exact_sol).cwiseAbs().maxCoeff();
    l2_errors.push_back(l2_err);
    linf_errors.push_back(linf_err);

    // Libera eval che non serve più
    eval.resize(0);
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
  Scalar f_greedy_threshold = 1e-2;
  int batch_size = 500;

  runMultigridTest(nu, f_greedy_threshold, batch_size);

  return 0;
}
