#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/KernelInterpolation"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2

using namespace FMCA;

// Franke Function
Scalar FrankeFunction(Scalar x, Scalar y) {
  Scalar term1 =
      (3.0 / 4.0) * std::exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0) / 4.0) -
                             ((9.0 * y - 2.0) * (9.0 * y - 2.0) / 4.0));
  Scalar term2 =
      (3.0 / 4.0) * std::exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0) / 49.0) -
                             ((9.0 * y - 2.0) * (9.0 * y - 2.0) / 10.0));
  Scalar term3 =
      (1.0 / 2.0) * std::exp(-((9.0 * x - 7.0) * (9.0 * x - 7.0) / 4.0) -
                             ((9.0 * y - 3.0) * (9.0 * y - 3.0) / 4.0));
  Scalar term4 = (1.0 / 5.0) * std::exp(-((9.0 * x - 4.0) * (9.0 * x - 4.0)) -
                                        ((9.0 * y - 7.0) * (9.0 * y - 7.0)));
  return term1 + term2 + term3 - term4;
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

// Summable weight c_m of the sum kernel S_l = sum_{m<=l} c_m k_m.
Scalar levelWeight(int m) { return 1.0 / Scalar(m + 1); }  // c_m = 1/(m+1)

//////////////////////////////////////////////////////////////////////////////////////////
template <typename KernelSolver>
void runMultigridTest(Scalar nu) {
  ////////////////////////////// Points
  std::vector<int> gridSizes = {4, 9, 25, 81, 289, 1089, 4225, 16641, 66049, 262145, 1048577};
  // quick check: {9, 25, 81, 289, 1089, 4225};
  std::vector<Matrix> P_levels;
  for (int size : gridSizes) {
    P_levels.push_back(generateUniformGrid(size));
  }
  Matrix Peval = generateUniformGrid(40000);

  ////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar threshold = 1e-6;
  const std::string kernel_type = "matern32";
  const Scalar ridgep = 0;
  const bool preconditioner = true;
  Scalar cg_threshold = 1e-6;

  ////////////////////////////// Multiscale Interpolator
  MultiscaleInterpolator<KernelSolver> MSI;
  MSI.init(P_levels, dtilde, eta, threshold, ridgep, nu, DIM);
  const Index num_levels = MSI.numLevels();

  ////////////////////////////// Residuals
  std::vector<Vector> residuals(num_levels);
  for (int l = 0; l < num_levels; ++l) {
    residuals[l] = evalFrankeFunction(MSI.points(l));
  }

  ////////////////////////////// Vector of coefficients (solution at each level)
  std::vector<Vector> ALPHA(num_levels);

  ////////////////////////////// Summary results
  std::vector<int> levels;
  std::vector<int> N;
  std::vector<Scalar> compression_time;
  std::vector<Scalar> cg_time;
  std::vector<int> iterationsCG;

  ////////////////////////////// Diagonal Loop
  Tictoc timer;
  for (int l = 0; l < num_levels; ++l) {
    std::cout << "Level " << l << std::endl;

    ////////////////////////////// Extra-Diagonal Loop
    // residual_l -= sum_{j<l} S_j(X_l, X_j) z_j , using the sum kernel S_j.
    for (int j = 0; j < l; ++j) {
      // build the sum kernel S_j = sum_{m<=j} c_m * sigma_m^{-DIM} * k(.;sigma_m)
      std::vector<Scalar> scales, weights;
      for (int m = 0; m <= j; ++m) {
        Scalar sigma_m = nu * MSI.fillDistance(m);
        scales.push_back(sigma_m);
        weights.push_back(levelWeight(m) * std::pow(sigma_m, -DIM));
      }
      SumKernel kernel_j(kernel_type, scales, weights);

      MultipoleFunctionEvaluator evaluator;
      evaluator.init(kernel_j, MSI.points(j), MSI.points(l));
      Matrix correction =
          evaluator.evaluate(MSI.points(j), MSI.points(l), ALPHA[j]);
      residuals[l] -= correction;
    }

    ////////////////////////////// Compress the diagonal block (sum kernel S_l)
    Scalar compress_time;
    {  // SCOPE BLOCK
      // build the sum kernel S_l = sum_{m<=l} c_m * sigma_m^{-DIM} * k(.;sigma_m)
      std::vector<Scalar> scales, weights;
      for (int m = 0; m <= l; ++m) {
        Scalar sigma_m = nu * MSI.fillDistance(m);
        scales.push_back(sigma_m);
        weights.push_back(levelWeight(m) * std::pow(sigma_m, -DIM));
      }
      SumKernel kernel_l(kernel_type, scales, weights);

      timer.tic();
      MSI.solver(l).compress(MSI.points(l), kernel_l);
      compress_time = timer.toc();
    }  // kernel_l destroyed HERE (compression already done)

    ////////////////////////////// Solver  (S_l z_l = residual_l)
    // Weights are baked into S_l, so the solution is already the final
    // coefficient (no sigma^{-DIM} post-scaling needed).
    Scalar solver_time;
    {  // SCOPE BLOCK
      timer.tic();
      ALPHA[l] = MSI.solver(l).solveIteratively(residuals[l], preconditioner,
                                                cg_threshold);
      solver_time = timer.toc();
    }

    levels.push_back(l + 1);
    N.push_back(MSI.points(l).cols());
    compression_time.push_back(compress_time);
    cg_time.push_back(solver_time);
    iterationsCG.push_back(MSI.solver(l).solver_iterations());
  }

  ////////////////////////////// Evaluation
  std::cout << "Evaluation... " << std::endl;
  Vector exact_sol = evalFrankeFunction(Peval);
  Vector final_res = Vector::Zero(Peval.cols());
  std::vector<Scalar> l2_errors;
  std::vector<Scalar> linf_errors;
  for (int l = 0; l < num_levels; ++l) {
    // build the sum kernel S_l = sum_{m<=l} c_m * sigma_m^{-DIM} * k(.;sigma_m)
    std::vector<Scalar> scales, weights;
    for (int m = 0; m <= l; ++m) {
      Scalar sigma_m = nu * MSI.fillDistance(m);
      scales.push_back(sigma_m);
      weights.push_back(levelWeight(m) * std::pow(sigma_m, -DIM));
    }
    SumKernel kernel_l(kernel_type, scales, weights);

    MultipoleFunctionEvaluator evaluator;
    evaluator.init(kernel_l, MSI.points(l), Peval);
    Vector eval = evaluator.evaluate(MSI.points(l), Peval, ALPHA[l]);
    final_res += eval;
    // Errors
    Scalar l2_err = (final_res - exact_sol).norm() / exact_sol.norm();
    Scalar linf_err = (final_res - exact_sol).cwiseAbs().maxCoeff();
    l2_errors.push_back(l2_err);
    linf_errors.push_back(linf_err);
  }

  ////////////////////////////// Results Summary
  std::cout << "\n======== Results Summary (sum kernel, c_m = 1/(m+1)) ========"
            << std::endl;
  std::cout << std::left << std::setw(10) << "Level" << std::setw(10) << "N"
            << std::setw(15) << "CompressTime" << std::setw(15) << "CGTime"
            << std::setw(10) << "IterCG" << std::setw(15) << "L2 Error"
            << std::setw(20) << "Linf Error" << std::endl;
  for (size_t i = 0; i < levels.size(); ++i) {
    std::cout << std::left << std::setw(10) << levels[i] << std::setw(10)
              << N[i] << std::setw(15) << std::fixed << std::setprecision(6)
              << compression_time[i] << std::setw(15) << std::fixed
              << std::setprecision(6) << cg_time[i] << std::setw(10)
              << iterationsCG[i] << std::setw(15) << std::scientific
              << std::setprecision(6) << l2_errors[i] << std::setw(20)
              << std::scientific << std::setprecision(6) << linf_errors[i]
              << std::endl;
  }
}

////////////////////////////// MAIN
int main() {
  // Test different nu values
  std::vector<Scalar> nus = {1.0};

  for (Scalar nu : nus) {
    runMultigridTest<SampletKernelSolver<>>(nu);
    std::cout << "\n\n";
  }

  return 0;
}