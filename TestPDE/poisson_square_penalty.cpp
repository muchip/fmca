/* We solve here the Poisson problem on the square [-1,1]x[-1,1]
laplacian(u) = 2*pi^2*sin(pi*x)*sin(pi*y)
u = 0 on \partial u
Using Matern32 kernel and penalty method to impose the boundary conditions.
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
  int NPTS_QUAD;
  int NPTS_QUAD_BORDER;
  int N_WEIGHTS;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_sources_border;
  FMCA::Matrix P_quad_border;
  FMCA::Vector w_vec;
  //////////////////////////////////////////////////////////////////////////////
  readTXT("data/vertices_square.txt", P_sources, NPTS_SOURCE, 2);
  readTXT("data/quadrature7_points_square.txt", P_quad, NPTS_QUAD, 2);
  readTXT("data/quadrature7_weights_square.txt", w_vec, N_WEIGHTS);
  readTXT("data/sources_border.txt", P_sources_border, NPTS_SOURCE_BORDER, 2);
  readTXT("data/baricenters_border.txt", P_quad_border, NPTS_QUAD_BORDER, 2);
  //////////////////////////////////////////////////////////////////////////////
  FMCA::Vector w_vec_border(NPTS_QUAD_BORDER);
  w_vec_border.setOnes();
  w_vec_border = w_vec_border * 0.0625/2;
  //////////////////////////////////////////////////////////////////////////////

//   FMCA::Vector indices_left;
//   FMCA::Vector indices_right;
//   FMCA::Vector indices_top;
//   FMCA::Vector indices_bottom;
//   for (int i = 0; i < P_sources.cols(); ++i) {
//     if (P_sources(0, i) == -1) {
//       indices_left.conservativeResize(indices_left.size() + 1);
//       indices_left(indices_left.size() - 1) = i;
//     } else if (P_sources(0, i) == 1) {
//       indices_right.conservativeResize(indices_right.size() + 1);
//       indices_right(indices_right.size() - 1) = i;
//     } else if (P_sources(1, i) == 1 & P_sources(0, i) != -1 &
//                P_sources(0, i) != 1) {
//       indices_top.conservativeResize(indices_top.size() + 1);
//       indices_top(indices_top.size() - 1) = i;
//     } else if (P_sources(1, i) == -1 & P_sources(0, i) != -1 &
//                P_sources(0, i) != 1) {
//       indices_bottom.conservativeResize(indices_bottom.size() + 1);
//       indices_bottom(indices_bottom.size() - 1) = i;
//     }
//   }
  // Boundary sources points
//   NPTS_SOURCE_BORDER = indices_left.size() + indices_right.size() +
//                        indices_top.size() + indices_bottom.size();
  // U_BC vector
  FMCA::Vector u_bc(NPTS_QUAD_BORDER);
  for (int i = 0; i < NPTS_QUAD_BORDER; ++i) {
    double x = P_quad_border(0, i);
    double y = P_quad_border(1, i);
    u_bc[i] = cos(FMCA_PI*x)*cos(FMCA_PI*y);
  }

  // Right hand side f = 0
  FMCA::Vector f(NPTS_QUAD);
  for (int i = 0; i < NPTS_QUAD; ++i) {
    double x = P_quad(0, i);
    double y = P_quad(1, i);
    f[i] =  2 * FMCA_PI * FMCA_PI * cos(FMCA_PI * x) * cos(FMCA_PI * y);
  }
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 4;
  const FMCA::Scalar threshold = 1e-6;
  const FMCA::Scalar MPOLE_DEG = 6;
  const FMCA::Scalar sigma = 0.1;
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
  const FMCA::CovarianceKernel kernel_funtion("MATERN32", sigma);
  Eigen::MatrixXd Kernel = kernel_funtion.eval(P_sources, P_quad);
  //////////////////////////////////////////////////////////////////////////////
  //   std::cout << w_vec_border.size() << std::endl;
  //   std::cout << P_quad_border.size() << std::endl;

  const FMCA::CovarianceKernel kernel_funtion_border("MATERN32", sigma);
  Eigen::MatrixXd Kernel_border =
      kernel_funtion_border.eval(P_sources, P_quad_border);
  Eigen::MatrixXd penalty =
      Kernel_border *
      (w_vec_border.asDiagonal() * Kernel_border.transpose()).eval();
  // FMCA::IO::print2m("Kernel_square.m", "Kernel_square_matrix", mass, "w");
  //////////////////////////////////////////////////////////////////////////////

  Eigen::MatrixXd stiffness(NPTS_SOURCE, NPTS_SOURCE);
  stiffness.setZero();

  for (int i = 0; i < DIM; ++i) {
    std::cout << std::string(80, '-') << std::endl;
    const FMCA::GradKernel function("MATERN32", sigma, 1, i);
    Eigen::MatrixXd gradKernel = function.eval(P_sources, P_quad);
    Eigen::MatrixXd gradk =
        gradKernel * (w_vec.asDiagonal() * gradKernel.transpose()).eval();
    stiffness += gradk;
  }
  // FMCA::IO::print2m("stiffness.m", "stiffness_matrix", stiffness, "w");
  // FMCA::IO::print2m("mass.m", "mass_matrix", mass, "w");
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // F_comp = right hand side
  FMCA::Vector rhs = Kernel * (w_vec.asDiagonal() * f).eval();
  //////////////////////////////////////////////////////////////////////////////
  // G_comp = right hand side
  FMCA::Vector rhs1 = Kernel_border * (w_vec_border.asDiagonal() * u_bc).eval();
  //////////////////////////////////////////////////////////////////////////////
  // solve
  Eigen::FullPivHouseholderQR<Eigen::MatrixXd> solver;
  solver.compute(stiffness + 1000 * penalty);
  FMCA::Vector numerical_sol = solver.solve(rhs + 1000 * rhs1);
  FMCA::Vector KU = kernel_funtion.eval(P_sources, P_sources) * numerical_sol;
  FMCA::Vector analytical_sol(NPTS_SOURCE);
  for (int i = 0; i < NPTS_SOURCE; ++i) {
    double x = P_sources(0, i);
    double y = P_sources(1, i);
    analytical_sol[i] = cos(FMCA_PI * x) * cos(FMCA_PI * y);
  }
  std::cout << "residual error:    "
            << ((stiffness + 1000 * penalty) * numerical_sol - (rhs+ 1000 * rhs1)).norm()
            << std::endl;
  std::cout << "(KU - analytical_sol).norm() / analytical_sol.norm():    "
            << (KU - analytical_sol).norm() / analytical_sol.norm()
            << std::endl;
  FMCA::Vector absolute_error(NPTS_SOURCE);
  for (int i = 0; i < NPTS_SOURCE; ++i) {
    absolute_error[i] = abs(KU[i] - analytical_sol[i]);
  }
  FMCA::Vector relative_error(NPTS_SOURCE);
  for (int i = 0; i < NPTS_SOURCE; ++i) {
    relative_error[i] = abs(KU[i] - analytical_sol[i])/abs(analytical_sol[i]);
  }
  FMCA::IO::print2m("solution_square_penalty.m", "sol_square_penalty", KU, "w");
  FMCA::IO::print2m("error_square_penalty_absolute.m", "absolute_error_penalty",
                    absolute_error, "w");
  FMCA::IO::print2m("error_square_penalty_relative.m", "relative_error_penalty",
                    relative_error, "w");

  return 0;
}
