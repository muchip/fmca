/* We solve here the Poisson problem on the square [-1,1]x[-1,1]
laplacian(u) = 2*pi^2*sin(pi*x)*sin(pi*y)
u = 0 on \partial u
Using Matern32 kernel and penalty method to impose the boundary conditions.
We rely on the FMCA library by M.Multerer.
 */

#include "SolvePoisson.h"
#include "read_files_txt.h"
#include "../FMCA/src/util/Plotter.h"

#define DIM 2

int main() {
  // Points
  FMCA::Tictoc T;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_quad_border;
  FMCA::Matrix Peval;
  FMCA::Matrix Normals;
  FMCA::Vector w_vec;
  FMCA::Vector w_vec_border;

  readTXT("data/uniform_vertices_square10k.txt", P_sources, 2);
  // readTXT("data/vertices_square_Halton50000.txt", P_sources, 2);

  readTXT("data/inside_points_square_MC.txt", P_quad, 2);
  readTXT("data/inside_weights_square_MC.txt", w_vec);
  // readTXT("data/quadrature3_points_square90k.txt", P_quad, 2);
  // readTXT("data/quadrature3_weights_square90k.txt", w_vec);

  readTXT("data/quadrature_border40k.txt", P_quad_border, 2);
  readTXT("data/weights_border40k.txt", w_vec_border);
  readTXT("data/normals40k.txt", Normals, 2);
  readTXT("data/uniform_vertices_square10k.txt", Peval, 2);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // U_BC vector
  FMCA::Vector u_bc(P_quad_border.cols());
  for (FMCA::Index i = 0; i < P_quad_border.cols(); ++i) {
    FMCA::Scalar x = P_quad_border(0, i);
    FMCA::Scalar y = P_quad_border(1, i);
    u_bc[i] = 0;
  }
  // Right hand side
  FMCA::Vector f(P_quad.cols());
  for (FMCA::Index i = 0; i < P_quad.cols(); ++i) {
    FMCA::Scalar x = P_quad(0, i);
    FMCA::Scalar y = P_quad(1, i);
    f[i] = 2 * FMCA_PI * FMCA_PI * sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }
  // Analytical sol of the problem
  FMCA::Vector analytical_sol(P_sources.cols());
  for (int i = 0; i < P_sources.cols(); ++i) {
    double x = P_sources(0, i);
    double y = P_sources(1, i);
    analytical_sol[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 1. / DIM;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar threshold_kernel = 1e-6;
  const FMCA::Scalar threshold_gradKernel = 1e-2;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 4;
  const FMCA::Scalar beta = 10000;
  const std::string kernel_type = "MATERN32";
  std::string outputNamePlot = "Galerkin.vtk";
  std::string outputNameErrorPlot = "error_Galerkin.vtk";

  const Moments mom_sources(P_sources, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  FMCA::Vector minDistance = minDistanceVector(hst_sources, P_sources);
  FMCA::Scalar sigma_h = minDistance.mean();
  std::cout << "average distance:                    " << sigma_h << std::endl;
  FMCA::Scalar sigma = 2 * sigma_h;

  FMCA::Vector u = SolvePoisson_constantWeighs(
      2, P_sources, P_quad, w_vec(0), P_quad_border, w_vec_border(0), Normals, u_bc,
      f, sigma, eta, dtilde, threshold_kernel, threshold_gradKernel,
      threshold_weights, MPOLE_DEG, beta, kernel_type);

  FMCA::Vector u_grid = hst_sources.inverseSampletTransform(u);
  u_grid = hst_sources.toNaturalOrder(u_grid);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // plot solution
  {
    const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
    const Moments cmom(P_sources, MPOLE_DEG);
    const Moments rmom(Peval, MPOLE_DEG);
    const H2ClusterTree hct(cmom, 0, P_sources);
    const H2ClusterTree hct_eval(rmom, 0, Peval);
    const FMCA::Vector sx = hct.toClusterOrder(u_grid);
    const usMatrixEvaluatorKernel mat_eval(rmom, cmom, kernel_funtion_ss);
    FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster> hmat;
    hmat.computePattern(hct_eval, hct, eta);
    hmat.statistics();
    FMCA::Vector srec = hmat.action(mat_eval, sx);
    FMCA::Vector rec = hct_eval.toNaturalOrder(srec);
    // FMCA::Plotter2D plotter;
    // plotter.plotFunction(outputNamePlot, Peval, rec);

    // Analytical sol of the problem
    FMCA::Vector analytical_sol(Peval.cols());
    for (int i = 0; i < Peval.cols(); ++i) {
      double x = Peval(0, i);
      double y = Peval(1, i);
      analytical_sol[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
    }

    FMCA::Scalar error = (rec - analytical_sol).norm() / analytical_sol.norm();
    std::cout << "Error:                                            " << error
              << std::endl;

    // FMCA::Vector absolute_error(Peval.cols());
    // for (int i = 0; i < Peval.cols(); ++i) {
    //   absolute_error[i] = abs(rec[i] - analytical_sol[i]);
    // }

    // // plot error
    // {
    //   FMCA::Plotter2D plotter_error;
    //   plotter_error.plotFunction(outputNameErrorPlot, Peval, absolute_error);
    // }
    }
  return 0;
}
