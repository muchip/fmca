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

//////////////////////////////////////////////////////////////////////////////////////////
template <typename KernelSolver>
void runMultigridTest(Scalar nu) {
  ////////////////////////////// Points
  std::vector<int> gridSizes = {9,    25,    81,    289,    1089,
                                4225, 16641, 66049, 262145};
  std::vector<Matrix> P_levels;
  for (int size : gridSizes) {
    P_levels.push_back(generateUniformGrid(size));
  }
  Matrix Peval = generateUniformGrid(40000);

  ////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar threshold = 1e-6;
  const Scalar mpole_deg = 2 * (dtilde - 1);
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
  std::vector<Scalar> fill_distances(num_levels);

  ////////////////////////////// Summary results
  std::vector<int> levels;
  std::vector<int> N;
  std::vector<Scalar> compression_time;
  std::vector<Scalar> cg_time;
  std::vector<int> iterationsCG;
  std::vector<size_t> anz;

  ////////////////////////////// Diagonal Loop
  Tictoc timer;
  for (int l = 0; l < num_levels; ++l) {
    std::cout << "Level " << l << std::endl;

    ////////////////////////////// Extra-Diagonal Loop
    for (int j = 0; j < l; ++j) {
      {  // SCOPE BLOCK 
        Scalar sigma_B = nu * MSI.fillDistance(j);
        CovarianceKernel kernel_B(kernel_type, sigma_B);
        MultipoleFunctionEvaluator evaluator;
        evaluator.init(kernel_B, MSI.points(j), MSI.points(l));
        Matrix correction =
            evaluator.evaluate(MSI.points(j), MSI.points(l), ALPHA[j]);
        correction *= std::pow(sigma_B, -DIM);
        residuals[l] -= correction;
      }  // sigma_B, kernel_B, evaluator, correction all destroyed HERE
    }

    ////////////////////////////// Compress the diagonal block
    Scalar compress_time;
    {  // SCOPE BLOCK 
      Scalar sigma_l = nu * MSI.fillDistance(l);
      CovarianceKernel kernel_l(kernel_type, sigma_l);
      timer.tic();
      MSI.solver(l).compress(MSI.points(l), kernel_l);
      compress_time = timer.toc();
      // Scalar comp_error = MSI.solver(l).compressionError(MSI.points(l),
      // kernel_l);
    }  // sigma_l, kernel_l destroyed HERE

    ////////////////////////////// Solver
    Scalar solver_time;
    {  // SCOPE BLOCK 
      Scalar sigma_l = nu * MSI.fillDistance(l);
      timer.tic();
      Vector solution = MSI.solver(l).solveIteratively(
          residuals[l], preconditioner, cg_threshold);
      solver_time = timer.toc();

      solution /= std::pow(sigma_l, -DIM);
      ALPHA[l] = solution;
    }  // solution, sigma_l destroyed HERE

    levels.push_back(l + 1);
    N.push_back(MSI.points(l).cols());
    compression_time.push_back(compress_time);
    cg_time.push_back(solver_time);
    iterationsCG.push_back(MSI.solver(l).solver_iterations());
  }

  ////////////////////////////// Evaluation
  std::cout << "Evaluation... "<< std::endl;
  Vector exact_sol = evalFrankeFunction(Peval);
  Vector final_res = Vector::Zero(Peval.cols());
  std::vector<Scalar> l2_errors;
  std::vector<Scalar> linf_errors;
  for (int l = 0; l < num_levels; ++l) {
    Scalar sigma = nu * MSI.fillDistance(l);
    CovarianceKernel kernel(kernel_type, sigma);
    MultipoleFunctionEvaluator evaluator;
    evaluator.init(kernel, MSI.points(l), Peval);
    Vector eval = evaluator.evaluate(MSI.points(l), Peval, ALPHA[l]);
    eval *= std::pow(sigma, -DIM);
    final_res += eval;
    // Errors
    Scalar l2_err = (final_res - exact_sol).norm() / exact_sol.norm();
    Scalar linf_err = (final_res - exact_sol).cwiseAbs().maxCoeff();
    l2_errors.push_back(l2_err);
    linf_errors.push_back(linf_err);
  }

  ////////////////////////////// Results Summary
  std::cout << "\n======== Results Summary ========" << std::endl;
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
    runMultigridTest<H2MatrixKernelSolver>(nu);
  }

  return 0;
}