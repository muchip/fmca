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
#include "Uzawa.h"
#include "read_files_txt.h"

#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
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
  FMCA::Matrix P_quad_border;
  FMCA::Vector w_vec;
  //////////////////////////////////////////////////////////////////////////////
  readTXT("data/vertices_circle.txt", P_sources, NPTS_SOURCE, 2);
  readTXT("data/quadrature7_points_circle.txt", P_quad, NPTS_QUAD, 2);
  readTXT("data/quadrature7_weights_circle.txt", w_vec, N_WEIGHTS);
  //////////////////////////////////////////////////////////////////////////////
  // Right hand side f = 0
  FMCA::Vector f(NPTS_QUAD);
  for (int i = 0; i < NPTS_QUAD; ++i) {
    double x = P_quad(0, i);
    double y = P_quad(1, i);
    f[i] = exp(-1.0 / 10 * sqrt(x * x + y * y));
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
  //   Weights matrix
  FMCA::Vector w_perm = w_vec(permutationVector(hst_quad));
  Eigen::MatrixXd W(w_vec.size(), w_vec.size());
  for (int i = 0; i < w_vec.size(); ++i) {
    W(i, i) = w_perm(i);
  }
  // P_sources.row(0)(permutationVector(hst_sources));
  // P_sources.row(1)(permutationVector(hst_sources));
  // P_quad.row(0)(permutationVector(hst_quad));
  // P_quad.row(1)(permutationVector(hst_quad));
  //////////////////////////////////////////////////////////////////////////////
  const FMCA::CovarianceKernel kernel_funtion("MATERN32", sigma);
  Eigen::MatrixXd Kernel = kernel_funtion.eval(P_sources, P_quad);
  Eigen::MatrixXd mass = Kernel * (W * Kernel.transpose()).eval();
  //////////////////////////////////////////////////////////////////////////////
  FMCA::IO::print2m("mass_circle.m", "mass_matrix_circle", mass, "w");
  //////////////////////////////////////////////////////////////////////////////
  // F_comp = right hand side 1
  FMCA::Vector rhs = Kernel * (W * f).eval();
  std::cout << "F_comp created" << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  std::cout << "determinant mass: " << Eigen::MatrixXd(mass).determinant()
            << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // // solve
  Eigen::LLT<Eigen::MatrixXd> solver;
  solver.compute(mass);
  FMCA::Vector numerical_sol = solver.solve(rhs);
  std::cout << "numerical sol done" << std::endl;
  FMCA::Vector KU = kernel_funtion.eval(P_sources, P_sources) * numerical_sol;
  FMCA::Vector analytical_sol(NPTS_SOURCE);
  for (int i = 0; i < NPTS_SOURCE; ++i) {
    double x = P_quad(0, i);
    double y = P_quad(1, i);
    analytical_sol[i] = exp(-1.0 / 10 * sqrt(x * x + y * y));
  }
  std::cout << "residual error:    " << ((mass)*numerical_sol - rhs).norm()
            << std::endl;
  std::cout << "error:    "
            << (KU - analytical_sol).squaredNorm() /
                   analytical_sol.squaredNorm()
            << std::endl;
  FMCA::IO::print2m("solution_mass_circle.m", "sol_circle_matrix", KU, "w");
  FMCA::IO::print2m("analytical_solution_mass_circle.m",
                    "analytical_sol_circle_matrix", analytical_sol, "w");
  return 0;
}
