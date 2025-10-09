#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/KernelInterpolation"
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
template<typename KernelSolver>
void runMultigridTest(Scalar nu) {
  std::cout << "======== Running Multigrid Test with nu = " << nu
            << " ========" << std::endl;

  ////////////////////////////// Points
  std::vector<int> gridSizes = {9, 25, 81, 289, 1089, 4225, 16641, 66049};
  std::vector<Matrix> P_levels;
  for (int size : gridSizes) {
    P_levels.push_back(generateUniformGrid(size));
  }
  Matrix Peval = generateUniformGrid(40000);

  ////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar threshold = 1e-8;
  const Scalar mpole_deg = 2 * (dtilde - 1);
  const std::string kernel_type = "matern52";
  const Scalar ridgep = 0;
  const bool preconditioner = true;
  Scalar cg_threshold = 1e-6;

  std::cout << "Parameters:" << std::endl;
  std::cout << "- eta:              " << eta << std::endl;
  std::cout << "- dtilde:           " << dtilde << std::endl;
  std::cout << "- threshold a post: " << threshold << std::endl;
  std::cout << "- mpole_deg:        " << mpole_deg << std::endl;
  std::cout << "- kernel_type:      " << kernel_type << std::endl;
  std::cout << "- nu:               " << nu << std::endl;

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
  std::vector<Scalar> assembly_time;  
  std::vector<Scalar> cg_time;        
  std::vector<int> iterationsCG;      
  std::vector<Scalar> anz;

  ////////////////////////////// Diagonal Loop
  for (int l = 0; l < num_levels; ++l) {
    std::cout << std::endl;
    std::cout << "-------- LEVEL " << (l + 1) << " --------" << std::endl;
    std::cout << "Fill distance: " << MSI.fillDistance(l) << std::endl;
    std::cout << "Number of points: " << MSI.points(l).cols() << std::endl;

    ////////////////////////////// Extra-Diagonal Loop
    for (int j = 0; j < l; ++j) {
      Scalar sigma_B = nu * MSI.fillDistance(j);
      CovarianceKernel kernel_B(kernel_type, sigma_B);
      MultipoleFunctionEvaluator evaluator;
      evaluator.init(kernel_B, MSI.points(j), MSI.points(l));
      Matrix correction =
          evaluator.evaluate(MSI.points(j), MSI.points(l), ALPHA[j]);
      correction *= std::pow(sigma_B, -DIM);
      residuals[l] -= correction;
    }

    ////////////////////////////// Compress the diagonal block
    Scalar sigma_l = nu * MSI.fillDistance(l);
    CovarianceKernel kernel_l(kernel_type, sigma_l);
    MSI.solver(l).compress(MSI.points(l), kernel_l);

    ////////////////////////////// Statistics
    const auto& compressor_stats = MSI.solver(l).getCompressionStats();
    std::cout << "\nCompression Stats:" << std::endl;
    std::cout << "- Compression time: " << compressor_stats.time_compressor
              << " s" << std::endl;
    // std::cout << "- Assembly time:    " << compressor_stats.assembly_time
    //           << std::endl;
    // std::cout << "- ANZ (non-zeros per row): " << compressor_stats.anz
    //           << std::endl;
    std::cout << "- Compression error: " << compressor_stats.compression_error
              << std::endl;

    ////////////////////////////// Solver 
    Vector solution =
        MSI.solver(l).solveIteratively(residuals[l], preconditioner, cg_threshold);
    solution /= std::pow(sigma_l, -DIM);
    ALPHA[l] = solution;
    
    //// Stats
    const auto& cg_stats = MSI.solver(l).getSolverStats();
    std::cout << "Solver Stats:" << std::endl;
    std::cout << "- Iterations: " << cg_stats.iterations << std::endl;
    std::cout << "- Residual: " << cg_stats.residual_error << std::endl;
    std::cout << "- Solver time: " << cg_stats.solver_time << " s" << std::endl;
    levels.push_back(l + 1);
    N.push_back(MSI.points(l).cols());
    assembly_time.push_back(compressor_stats.assembly_time);
    cg_time.push_back(cg_stats.solver_time);
    iterationsCG.push_back(cg_stats.iterations);
    anz.push_back(compressor_stats.anz);
  }

  ////////////////////////////// Evaluation
  std::cout << "\nEvaluating solution on " << Peval.cols() << " points..."
            << std::endl;
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
    std::cout << "======= Evaluation Level " << (l + 1)
              << " =======" << std::endl;
    std::cout << "- L2 error: " << l2_err << std::endl;
    std::cout << "- Linf error: " << linf_err << std::endl;
    std::cout << std::endl;
  }

  ////////////////////////////// Results Summary
  std::cout << "\n======== Results Summary ========" << std::endl;
  std::cout << std::left << std::setw(10) << "Level" << std::setw(10) << "N"
            << std::setw(15) << "AssemblyTime" << std::setw(15) << "CGTime"
            << std::setw(10) << "IterCG" << std::setw(10) << "ANZ"
            << std::setw(15) << "L2 Error" << std::setw(15) << "Linf Error"
            << std::endl;
  for (size_t i = 0; i < levels.size(); ++i) {
    std::cout << std::left << std::setw(10) << levels[i] << std::setw(10)
              << N[i] << std::setw(15) << std::fixed << std::setprecision(6)
              << assembly_time[i] << std::setw(15) << std::fixed
              << std::setprecision(6) << cg_time[i] << std::setw(10)
              << iterationsCG[i]
              << std::setw(10)
              << static_cast<int>(anz[i])
              << std::setw(15) << std::scientific << std::setprecision(6)
              << l2_errors[i] << std::setw(15) << std::scientific
              << std::setprecision(6) << linf_errors[i] << std::endl;
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