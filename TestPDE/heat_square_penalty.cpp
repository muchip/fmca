/* We solve here the heat problem on the square [0,1]x[0,1]
d u/ d t - laplacian(u) = exp(-pi^2*t)*sin(pi*x)*sin(pi*y)
u(x,0,t) = 0    u(x,1,t) = 0
u(y,0,t) = 0    u(y,1,t) = 0
u(x,y,0) = sin(pi*x)*sin(pi*y)

Using Matern32 kernel and penalty method to impose the boundary conditions.
We rely on the FMCA library by M.Multerer.
 */


// IT DOES NOT WORK

#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Eigenvalues>
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

FMCA::Vector compute_f(double t, int NPTS_QUAD, FMCA::Matrix P_quad) {
  FMCA::Vector f(NPTS_QUAD);
  for (int i = 0; i < NPTS_QUAD; ++i) {
    double x = P_quad(0, i);
    double y = P_quad(1, i);
    f[i] = -exp(-FMCA_PI * FMCA_PI * t) * sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }
  return f;
}

int main() {
  // Initialize the matrices of source points, quadrature points, weights
  FMCA::Tictoc T;
  int NPTS_SOURCE;
  int NPTS_SOURCE_BORDER;
  int NPTS_QUAD;
  int NPTS_QUAD_BORDER;
  int N_WEIGHTS;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_sources_border;
  FMCA::Matrix P_quad_border;
  FMCA::Vector w_vec;
  //////////////////////////////////////////////////////////////////////////////
  readTXT("data/vertices_square_01.txt", P_sources, NPTS_SOURCE, 2);
  readTXT("data/quadrature7_points_square_01.txt", P_quad, NPTS_QUAD, 2);
  readTXT("data/quadrature7_weights_square_01.txt", w_vec, N_WEIGHTS);
  readTXT("data/baricenters_border_01.txt", P_quad_border, NPTS_QUAD_BORDER, 2);
  //////////////////////////////////////////////////////////////////////////////
  FMCA::Vector w_vec_border(NPTS_QUAD_BORDER);
  w_vec_border.setOnes();
  w_vec_border = w_vec_border * 0.015625;
  //////////////////////////////////////////////////////////////////////////////
  // U_BC vector
  FMCA::Vector u_bc(NPTS_QUAD_BORDER);
  for (int i = 0; i < NPTS_QUAD_BORDER; ++i) {
    u_bc[i] = 0;
  }
  //////////////////////////////////////////////////////////////////////////////
  // initial condition
  FMCA::Vector u_0(NPTS_SOURCE);
  for (int i = 0; i < NPTS_SOURCE; ++i) {
    double x = P_sources(0, i);
    double y = P_sources(1, i);
    u_0[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 4;
  const FMCA::Scalar threshold = 1e-6;
  const FMCA::Scalar MPOLE_DEG = 6;
  const FMCA::Scalar sigma = 0.1;
  const FMCA::Scalar dt = 0.001;
  const FMCA::Scalar time_steps = 1000;
  const FMCA::Scalar beta = 1000;
  //////////////////////////////////////////////////////////////////////////////
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const Moments mom_quad(P_quad, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
  std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
  std::cout << "Number of weights = " << N_WEIGHTS << std::endl;
  std::cout << "minimum element:                     "
            << *std::min_element(w_vec.begin(), w_vec.end()) << std::endl;
  std::cout << "maximum element:                     "
            << *std::max_element(w_vec.begin(), w_vec.end()) << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "sigma = " << sigma << std::endl;
  std::cout << "eta = " << eta << std::endl;
  std::cout << "dtilde = " << dtilde << std::endl;
  std::cout << "threshold = " << threshold << std::endl;
  std::cout << "MPOLE_DEG = " << MPOLE_DEG << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // initial alpha
  const FMCA::CovarianceKernel kernel_ss("MATERN32", sigma);
  Eigen::MatrixXd Kernel_ss = kernel_ss.eval(P_sources, P_sources);
  FMCA::Vector alpha_0(NPTS_SOURCE);
  Eigen::FullPivHouseholderQR<Eigen::MatrixXd> solver_0;
  solver_0.compute(Kernel_ss);
  alpha_0 = solver_0.solve(u_0);
  std::cout << "K*alpha_0 - u_0:      "
            << ((Kernel_ss * alpha_0).eval() - u_0).norm() / (u_0.norm())
            << std::endl;
  FMCA::IO::print2m("alpha0.m", "initial_solution", Kernel_ss * alpha_0, "w");
  //////////////////////////////////////////////////////////////////////////////
  // mass
  const FMCA::CovarianceKernel kernel_funtion("MATERN32", sigma);
  Eigen::MatrixXd Kernel = kernel_funtion.eval(P_sources, P_quad);
  Eigen::MatrixXd mass =
      Kernel * (w_vec.asDiagonal() * Kernel.transpose()).eval();
    std::cout<< "some entries of the Kernel(source,quad):    "<< Kernel(0,0) << " " << Kernel(0,1) << " " << Kernel(0,2) << std::endl;
  // Computing eigenvalues of the matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mass);
  Eigen::VectorXd eivals = solver.eigenvalues();

  // Finding maximum and minimum eigenvalues
  double max_eigenvalue = eivals.maxCoeff();
  double min_eigenvalue = eivals.minCoeff();
 std::cout<< "max_eigenvalue:    "<<max_eigenvalue <<std::endl;
 std::cout<< "min_eigenvalue:    "<<min_eigenvalue <<std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // penalty
  const FMCA::CovarianceKernel kernel_funtion_border("MATERN32", sigma);
  Eigen::MatrixXd Kernel_border =
      kernel_funtion_border.eval(P_sources, P_quad_border);
  Eigen::MatrixXd penalty =
      Kernel_border *
      (w_vec_border.asDiagonal() * Kernel_border.transpose()).eval();
  //////////////////////////////////////////////////////////////////////////////
  // stiffness
  Eigen::MatrixXd stiffness(NPTS_SOURCE, NPTS_SOURCE);
  stiffness.setZero();
  for (int i = 0; i < DIM; ++i) {
    const FMCA::GradKernel function("MATERN32", sigma, 1, i);
    Eigen::MatrixXd gradKernel = function.eval(P_sources, P_quad);
    Eigen::MatrixXd gradk =
        gradKernel * (w_vec.asDiagonal() * gradKernel.transpose()).eval();
    stiffness += gradk;
  }
  std::cout << std::string(80, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // G_comp = right hand side
  FMCA::Vector rhs1 = Kernel_border * (w_vec_border.asDiagonal() * u_bc).eval();
  //////////////////////////////////////////////////////////////////////////////
  // solve
  FMCA::Vector alpha = alpha_0;
  Eigen::FullPivHouseholderQR<Eigen::MatrixXd> solver_t;
  solver_t.compute(mass);
  for (int i = 1; i <= 1; ++i) {
    FMCA::Vector f_current = compute_f(i * dt, NPTS_QUAD, P_quad);
    FMCA::Vector Space = solver_t.solve(
        + dt * (stiffness * alpha + beta * penalty * alpha).eval() +
        dt * Kernel * (w_vec.asDiagonal() * f_current).eval() +
        dt * beta * rhs1);
    alpha += Space;
  }
  FMCA::IO::print2m("alpha.m", "solution_alpha", Kernel_ss * alpha, "w");
  /*
//////////////////////////////////////////////////////////////////////////////
// analytical solution at t = 1 = final time
int final_time = dt * time_steps;
FMCA::Vector analytical_sol(NPTS_SOURCE);
for (int i = 0; i < NPTS_SOURCE; ++i) {
double x = P_sources(0, i);
double y = P_sources(1, i);
analytical_sol[i] = exp(-FMCA_PI * FMCA_PI * final_time) *
              sin(FMCA_PI * x) * sin(FMCA_PI * y);
}
FMCA::Vector KU = Kernel_ss * alpha;
std::cout << "(KU - analytical_sol).norm() / analytical_sol.norm():    "
  << (KU - analytical_sol).norm() / analytical_sol.norm()
  << std::endl;
FMCA::Vector absolute_error(NPTS_SOURCE);
for (int i = 0; i < NPTS_SOURCE; ++i) {
absolute_error[i] = abs(KU[i] - analytical_sol[i]);
}
FMCA::Vector relative_error(NPTS_SOURCE);
for (int i = 0; i < NPTS_SOURCE; ++i) {
relative_error[i] = abs(KU[i] - analytical_sol[i]) / abs(analytical_sol[i]);
}
FMCA::IO::print2m("solution_heat_penalty_01.m", "sol_heat_penalty_01", KU,
          "w");
FMCA::IO::print2m("error_heat_penalty_absolute_01.m",
          "absolute_error_heat_01", absolute_error, "w");
FMCA::IO::print2m("error_heat_penalty_relative_01.m",
          "relative_error_heat_01", relative_error, "w");
          */
  return 0;
}
