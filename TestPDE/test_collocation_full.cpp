/* We solve here the Poisson problem on the unit shpere
- laplacian(u) = 6
u = 0 on \partial u
Using Gaussian and Matern52 kernel and Kansa method collocation method.
The matrices are comrpessed using Samplets.
We rely on the FMCA library by M.Multerer.
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
/////////////////////////////////////////////////
#include "../FMCA/GradKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/Grid2D.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Plotter3D.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "read_files_txt.h"

#define DIM 3

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluatorKernel =
    FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluatorKernel =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;

int main() {
  // Initialize the matrices of source points, quadrature points, weights
  FMCA::Tictoc T;
  FMCA::Matrix P_interior_full;
  FMCA::Matrix P_bnd_full;

  readTXT("data/MC_Interior_Sphere_collocation.txt", P_interior_full, DIM);
  readTXT("data/MC_Bnd_Sphere_collocation.txt", P_bnd_full, DIM);

  int N_sources = 7000;
  int N_sources_border = 3000;
  FMCA::Matrix P_sources = P_interior_full.leftCols(N_sources);
  FMCA::Matrix P_sources_border = P_bnd_full.leftCols(N_sources_border);

  FMCA::Matrix P(DIM, N_sources + N_sources_border);
  P << P_sources, P_sources_border;
  FMCA::IO::plotPoints("P_sphere.vtk", P);

  FMCA::Matrix Peval = P;

  int NPTS_SOURCE = P_sources.cols();
  int NPTS_SOURCE_BORDER = P_sources_border.cols();
  int N = P.cols();

  // U_BC vector
  FMCA::Vector u_bc(NPTS_SOURCE_BORDER);
  for (int i = 0; i < NPTS_SOURCE_BORDER; ++i) {
    u_bc[i] = 0;
  }

  // Right hand side f
  FMCA::Vector f(NPTS_SOURCE);
  for (int i = 0; i < NPTS_SOURCE; ++i) {
    f[i] = -6;
  }
  // Exact solution
  FMCA::Vector u_exact(N);
  for (int i = 0; i < N; ++i) {
    double x = P(0, i);
    double y = P(1, i);
    double z = P(2, i);
    u_exact[i] = 1 - x * x - y * y - z * z;
  }
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar MPOLE_DEG = 2 * (dtilde - 1);
  std::string kernel_type = "MATERN52";
  std::string kernel_type_laplacian = "MATERN52_SECOND_DERIVATIVE";
  //////////////////////////////////////////////////////////////////////////////
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  const Moments mom_border(P_sources_border, MPOLE_DEG);
  const SampletMoments samp_mom_border(P_sources_border, dtilde - 1);
  H2SampletTree hst_sources_border(mom_border, samp_mom_border, 0,
                                   P_sources_border);
  const Moments mom(P, MPOLE_DEG);
  const SampletMoments samp_mom(P, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, P);
  FMCA::Vector minDistance = minDistanceVector(hst, P);
  FMCA::Scalar sigma_h = minDistance.mean();
  std::cout << "average distance:                   " << sigma_h << std::endl;

  FMCA::Scalar sigma = 5;
  // FMCA::Scalar sigma = 10 * sigma_h;
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Number of interior points = " << NPTS_SOURCE << std::endl;
  std::cout << "Number of border points = " << NPTS_SOURCE_BORDER << std::endl;
  std::cout << "Total number of points = " << N << std::endl;
  std::cout << "sigma = " << sigma << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  const FMCA::CovarianceKernel kernel_funtion_border(kernel_type, sigma);
  FMCA::Matrix K_bnd_n = kernel_funtion_border.eval(P_sources_border, P);
  //////////////////////////////////////////////////////////////////////////////
  FMCA::Matrix Laplacian(NPTS_SOURCE, N);
  Laplacian.setZero();
  for (int i = 0; i < DIM; ++i) {
    const FMCA::GradKernel laplacian_funtion(kernel_type_laplacian, sigma, 1,
                                             i);
    FMCA::Matrix Laplacian_Interior = laplacian_funtion.eval(P_sources, P);
    Laplacian += Laplacian_Interior;
  }
  std::cout << "Kernel and Laplacian created" << std::endl;

  FMCA::Matrix A(N, N);
  A << Laplacian, K_bnd_n;

  FMCA::Vector b(N);
  b << f, u_bc;

  std::cout << "RHS created" << std::endl;

  // Solver
  Eigen::ColPivHouseholderQR<FMCA::Matrix> solver;
  solver.compute(A);
  // u = solution in Samplet Basis
  FMCA::Vector alpha = solver.solve(b);

  std::cout << "residual error:                          "
            << (A * alpha - b).norm() << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  const FMCA::CovarianceKernel kernel_funtion_N(kernel_type, sigma);
  FMCA::Matrix K = kernel_funtion_N.eval(P, P);

  FMCA::Vector Kalpha = K * alpha;

  std::cout << "(Kalpha - u_exact).norm() / analytical_sol.norm():    "
            << (Kalpha - u_exact).norm() / u_exact.norm() << std::endl;

  std::cout << std::string(80, '-') << std::endl;
  return 0;
}
