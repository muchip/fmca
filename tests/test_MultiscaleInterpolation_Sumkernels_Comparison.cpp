#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
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

// ---------------------------------------------------------------------------
// Summable weight sequences c_l of the sum kernel to compare:
//   "1/(l+1)"  : in l^2 but not l^1                 (the requested default)
//   "1/(l+1)^2": faster decay, in l^1               (stronger summability)
//   "2^-l"     : geometric, in l^1 and l^2          (fastest decay)
//   "1"        : unweighted reference (NOT summable, for comparison only)
// ---------------------------------------------------------------------------
using WeightFun = std::function<Scalar(int)>;

//////////////////////////////////////////////////////////////////////////////
// Multiscale interpolation with a SUM KERNEL, for one given weight sequence cl.
// At level l the block kernel is the running weighted sum
//     S_l = sum_{m<=l} cl(m) * sigma_m^{-DIM} * k(.; sigma_m).
// Because S_l is itself a (radial) CovarianceKernel (class SumKernel), every
// level is built with a SINGLE compression pass -- much faster than compressing
// each component and summing sparse blocks.
//
// The MultiscaleInterpolator MSI is built ONCE in main and reused for every
// weight sequence (the points and samplet trees do not depend on the weights),
// so the expensive tree construction is not repeated.
template <typename KernelSolver>
void runSumKernelTest(MultiscaleInterpolator<KernelSolver>& MSI,
                      const Matrix& Peval, Scalar nu,
                      const std::string& kernel_type, bool preconditioner,
                      Scalar cg_threshold, const WeightFun& cl,
                      const std::string& wlabel) {
  const Index num_levels = MSI.numLevels();

  ////////////////////////////// Per-level scales and weights
  // sigma_m and w_m = cl(m) * sigma_m^{-DIM} depend only on m.
  std::vector<Scalar> sigma(num_levels), w(num_levels);
  for (int m = 0; m < num_levels; ++m) {
    sigma[m] = nu * MSI.fillDistance(m);
    w[m] = cl(m) * std::pow(sigma[m], -DIM);
  }

  ////////////////////////////// Residuals and coefficients
  std::vector<Vector> residuals(num_levels);
  for (int l = 0; l < num_levels; ++l)
    residuals[l] = evalFrankeFunction(MSI.points(l));
  std::vector<Vector> ALPHA(num_levels);

  ////////////////////////////// Summary results
  std::vector<int> levels, N, iterationsCG;
  std::vector<Scalar> compression_time, cg_time;
  std::vector<size_t> nnz_per_row;

  ////////////////////////////// Diagonal (level) loop
  Tictoc timer;
  for (int l = 0; l < num_levels; ++l) {
    std::cout << "[" << wlabel << "] Level " << l << std::endl;

    ////////////////////////////// Off-diagonal correction
    // residual_l -= sum_{j<l} S_j(X_l, X_j) z_j , using the sum kernel S_j.
    for (int j = 0; j < l; ++j) {
      std::vector<Scalar> sc(sigma.begin(), sigma.begin() + (j + 1));
      std::vector<Scalar> wt(w.begin(), w.begin() + (j + 1));
      SumKernel S_j(kernel_type, sc, wt);
      MultipoleFunctionEvaluator<> evaluator;
      evaluator.init(S_j, MSI.points(j), MSI.points(l));
      Matrix correction =
          evaluator.evaluate(MSI.points(j), MSI.points(l), ALPHA[j]);
      residuals[l] -= correction;
    }

    ////////////////////////////// Diagonal block: compress S_l once, solve
    std::vector<Scalar> sc(sigma.begin(), sigma.begin() + (l + 1));
    std::vector<Scalar> wt(w.begin(), w.begin() + (l + 1));
    SumKernel S_l(kernel_type, sc, wt);

    timer.tic();
    MSI.solver(l).compress(MSI.points(l), S_l);
    Scalar compress_time = timer.toc();

    // Weights are baked into S_l, so the solution is already the final
    // coefficient (no sigma^{-DIM} post-scaling needed).
    timer.tic();
    ALPHA[l] = MSI.solver(l).solveIteratively(residuals[l], preconditioner,
                                              cg_threshold);
    Scalar solver_time = timer.toc();

    levels.push_back(l + 1);
    N.push_back(MSI.points(l).cols());
    compression_time.push_back(compress_time);
    cg_time.push_back(solver_time);
    iterationsCG.push_back(MSI.solver(l).solver_iterations());
    nnz_per_row.push_back(
        std::round(MSI.solver(l).K().nonZeros() / double(MSI.points(l).cols())));
  }

  ////////////////////////////// Evaluation (incremental, per level)
  std::cout << "[" << wlabel << "] Evaluation... " << std::endl;
  Vector exact_sol = evalFrankeFunction(Peval);
  Vector final_res = Vector::Zero(Peval.cols());
  std::vector<Scalar> l2_errors, linf_errors;
  for (int l = 0; l < num_levels; ++l) {
    // s_l(Peval) = S_l(Peval, X_l) z_l
    std::vector<Scalar> sc(sigma.begin(), sigma.begin() + (l + 1));
    std::vector<Scalar> wt(w.begin(), w.begin() + (l + 1));
    SumKernel S_l(kernel_type, sc, wt);
    MultipoleFunctionEvaluator<> evaluator;
    evaluator.init(S_l, MSI.points(l), Peval);
    Vector eval = evaluator.evaluate(MSI.points(l), Peval, ALPHA[l]);
    final_res += eval;
    Scalar l2_err = (final_res - exact_sol).norm() / exact_sol.norm();
    Scalar linf_err = (final_res - exact_sol).cwiseAbs().maxCoeff();
    l2_errors.push_back(l2_err);
    linf_errors.push_back(linf_err);
  }

  ////////////////////////////// Results Summary
  std::cout << "\n======== Sum-kernel results (weights c_l = " << wlabel
            << ") ========" << std::endl;
  std::cout << std::left << std::setw(8) << "Level" << std::setw(10) << "N"
            << std::setw(15) << "BuildTime" << std::setw(13) << "CGTime"
            << std::setw(10) << "IterCG" << std::setw(10) << "nnz/row"
            << std::setw(16) << "L2 Error" << std::setw(16) << "Linf Error"
            << std::endl;
  for (size_t i = 0; i < levels.size(); ++i) {
    std::cout << std::left << std::setw(8) << levels[i] << std::setw(10) << N[i]
              << std::setw(15) << std::fixed << std::setprecision(4)
              << compression_time[i] << std::setw(13) << std::fixed
              << std::setprecision(4) << cg_time[i] << std::setw(10)
              << iterationsCG[i] << std::setw(10) << nnz_per_row[i]
              << std::setw(16) << std::scientific << std::setprecision(4)
              << l2_errors[i] << std::setw(16) << std::scientific
              << std::setprecision(4) << linf_errors[i] << std::endl;
  }
}

////////////////////////////// MAIN
int main() {
  const Scalar nu = 2.0;
  const std::string kernel_type = "matern32";
  const Scalar eta = 1. / DIM;
  const Index dtilde = 4;
  const Scalar threshold = 1e-6;
  const Scalar ridgep = 0;
  const bool preconditioner = false;
  const Scalar cg_threshold = 1e-8;

  ////////////////////////////// Points (built ONCE, reused for all weights)
  std::vector<int> gridSizes = {4,    9,     25,    81,     289,    1089,
                                4225, 16641, 66049, 262145};
  std::vector<Matrix> P_levels;
  for (int size : gridSizes) P_levels.push_back(generateUniformGrid(size));
  Matrix Peval = generateUniformGrid(25000);

  ////////////////////////////// Multiscale interpolator (trees built ONCE)
  MultiscaleInterpolator<SampletKernelSolver<>> MSI;
  MSI.init(P_levels, dtilde, eta, threshold, ridgep, nu, DIM);

  ////////////////////////////// Weight sequences to compare
  std::vector<std::pair<std::string, WeightFun>> weights = {
      // {"1/(l+1)", [](int l) { return 1.0 / Scalar(l + 1); }},
      {"1/(l+1)^2", [](int l) { return 1.0 / Scalar((l + 1) * (l + 1)); }},
      {"(l+1)^2", [](int l) { return Scalar((l + 1) * (l + 1)); }},
      {"2^-l", [](int l) { return std::pow(2.0, -l); }},
      {"2^l", [](int l) { return std::pow(2.0, l); }},
      {"1", [](int l) { return 1.0; }}};

  std::cout << "### Sum-kernel multiscale interpolation, nu = " << nu << " ###"
            << std::endl;
  for (const auto& wpair : weights) {
    runSumKernelTest(MSI, Peval, nu, kernel_type, preconditioner, cg_threshold,
                     wpair.second, wpair.first);
    std::cout << "\n";
  }
  return 0;
}
