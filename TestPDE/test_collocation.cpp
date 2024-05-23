// NOT FINISHED --> PROBLEM: DIMENSIONS DO NOT MATCH

/* We solve here the Poisson problem on the square [-1,1]x[-1,1]
laplacian(u) = 2*pi^2*sin(pi*x)*sin(pi*y)
u = 0 on \partial u
Using Gaussian kernel and Kansa method collocation method.
We rely on the FMCA library by M.Multerer.
 */

#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense>
#include <cstdlib>
#include <iostream>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "read_files_txt.h"

#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluatorKernel =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  // Initialize the matrices of source points, quadrature points, weights
  FMCA::Tictoc T;
  int NPTS_SOURCE;
  int NPTS_SOURCE_BORDER;
  int N;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_sources_border;
  FMCA::Matrix P;
  //////////////////////////////////////////////////////////////////////////////
  readTXT("data/interior_square.txt", P_sources, NPTS_SOURCE, 2);
  readTXT("data/boundary_square.txt", P_sources_border, NPTS_SOURCE_BORDER, 2);
  readTXT("data/InteriorAndBnd_square.txt", P, N, 2);

  // U_BC vector
  FMCA::Vector u_bc(NPTS_SOURCE_BORDER);
  for (int i = 0; i < NPTS_SOURCE_BORDER; ++i) {
    // double x = P_sources_border(0, i);
    // double y = P_sources_border(1, i);
    u_bc[i] = 0;
  }

  // Right hand side f
  FMCA::Vector f(NPTS_SOURCE);
  for (int i = 0; i < NPTS_SOURCE; ++i) {
    double x = P_sources(0, i);
    double y = P_sources(1, i);
    f[i] = -2 * FMCA_PI * FMCA_PI * sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }

  // Exact solution
  FMCA::Vector u_exact(N);
  for (int i = 0; i < N; ++i) {
    double x = P(0, i);
    double y = P(1, i);
    u_exact[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 4;
  const FMCA::Scalar threshold = 1e-6;
  const FMCA::Scalar MPOLE_DEG = 6;
  const FMCA::Scalar sigma = 0.3;
  const FMCA::Scalar c = 1;
  //////////////////////////////////////////////////////////////////////////////
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Number of interior points = " << NPTS_SOURCE << std::endl;
  std::cout << "Number of border points = " << NPTS_SOURCE_BORDER << std::endl;
  std::cout << "sigma = " << sigma << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  const FMCA::CovarianceKernel kernel_funtion_border("GAUSSIAN", sigma, c);
  Eigen::MatrixXd Kernel_border =
      kernel_funtion_border.eval(P_sources_border, P);
  //////////////////////////////////////////////////////////////////////////////
  Eigen::MatrixXd Laplacian(NPTS_SOURCE, N);
  for (int i = 0; i < DIM; ++i) {
    const FMCA::GradKernel laplacian_funtion("GAUSSIAN_SECOND_DERIVATIVE",
                                             sigma, c, i);
    Laplacian += laplacian_funtion.eval(P_sources, P);
  }

  // Concatenate Laplacian matrix and Kernel_border vertically
  Eigen::MatrixXd A(NPTS_SOURCE + NPTS_SOURCE_BORDER, N);
  A << Laplacian, Kernel_border;

  FMCA::IO::print2m("results_matlab/FullMatrixCollocation.mat", "FullCollocation",
                    A, "w"); 

  // Concatenate f vector and u_bc vector vertically
  Eigen::VectorXd b(NPTS_SOURCE + NPTS_SOURCE_BORDER);
  b << f, u_bc;

  // Solve the linear system A * alpha = b
  Eigen::VectorXd alpha = A.colPivHouseholderQr().solve(b);

  // Check if the solution is valid
  if ((A * alpha - b).norm() / b.norm() < 1e-6) {
    std::cout << "Solution found." << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  const FMCA::CovarianceKernel kernel_funtion("GAUSSIAN", sigma, c);
  Eigen::MatrixXd Kernel = kernel_funtion.eval(P, P);
  FMCA::Vector Kalpha = Kernel * alpha;

  std::cout << "(Kalpha - u_exact).norm() / analytical_sol.norm():    "
            << (Kalpha - u_exact).squaredNorm() / u_exact.squaredNorm()
            << std::endl;

  FMCA::IO::print2m("results_matlab/solution_collocation.m", "sol_collocation", Kalpha, "w");

    FMCA::Vector absolute_error(N);
  for (int i = 0; i < N; ++i) {
    absolute_error[i] = abs(Kalpha[i] - u_exact[i]);
  }
  FMCA::Vector relative_error(N);
  for (int i = 0; i < N; ++i) {
    relative_error[i] = abs(Kalpha[i] - u_exact[i])/abs(u_exact[i]);
  }

  FMCA::IO::print2m("results_matlab/error_collocation_absolute.m", "absolute_error_collocation",
                    absolute_error, "w");
  FMCA::IO::print2m("results_matlab/error_collocation_relative.m", "relative_error_collocation",
                    relative_error, "w");

  return 0;
}
