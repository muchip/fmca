/* We solve here the Poisson problem on the unit shpere
- laplacian(u) = 6
u = 0 on \partial u
Using Matern32 kernel and penalty method to impose the boundary conditions.
We rely on the FMCA library by M.Multerer.
 */

#include "SolvePoisson.h"
///////////////////////////////////
#include "../FMCA/src/util/Plotter.h"
#include "read_files_txt.h"

#define DIM 3

// for generating th point, see LaplacianSphereCollocation.cpp

FMCA::Scalar WeightInterior(int N, FMCA::Scalar r) {
  return (4. / 3.) * FMCA_PI * r * r * r *
         (1. / N);  // Monte Carlo integration weights: Volume*1/N 
}

FMCA::Scalar WeightBnd(int N, FMCA::Scalar r) {
  return  4. * FMCA_PI * r * r *
         (1. / N);  // Monte Carlo integration weights: Surface*1/N
}

int main() {
  // DATA
  FMCA::Tictoc T;
  FMCA::Matrix P_interior_full;
  FMCA::Matrix P_bnd_full;
  readTXT("data/MC_Interior_Sphere_collocation.txt", P_interior_full, DIM);
  readTXT("data/MC_Bnd_Sphere_collocation.txt", P_bnd_full, DIM);

  const int NPTS_QUAD_INTERIOR = 8000;
  const int NPTS_QUAD_BORDER = 5000;
  const int NPTS_EVAL = 1000;
  //////////////////////////////////////////////////////////////////////////////
  // Points
  FMCA::Matrix P_quad_interior = P_interior_full.leftCols(NPTS_QUAD_INTERIOR);
  FMCA::IO::plotPoints("P_sphere_interior.vtk", P_quad_interior);
  FMCA::Matrix P_quad_border = P_bnd_full.leftCols(NPTS_QUAD_BORDER);
  FMCA::IO::plotPoints("P_sphere_bnd.vtk", P_quad_border);

  FMCA::Matrix Peval = P_interior_full.rightCols(NPTS_EVAL);
  FMCA::IO::plotPoints("P_sphere_eval.vtk", Peval);
  FMCA::Matrix P_sources = Peval;

  FMCA::Matrix Normals = P_quad_border;

  //////////////////////////////////////////////////////////////////////////////
  // Weights
  FMCA::Scalar w = WeightInterior(NPTS_QUAD_INTERIOR, 1.0);
  std::cout << "w_vec:                      " << w << std::endl;
  FMCA::Vector w_vec = Eigen::VectorXd::Ones(NPTS_QUAD_INTERIOR);
  w_vec *= w;

  FMCA::Scalar w_border = WeightBnd(NPTS_QUAD_BORDER, 1.0);
  std::cout << "w_vec_border:               " << w_border << std::endl;
  FMCA::Vector w_vec_border = Eigen::VectorXd::Ones(NPTS_QUAD_BORDER);
  w_vec_border *= w_border;

  //////////////////////////////////////////////////////////////////////////////
  // U_BC vector
  FMCA::Vector u_bc(P_quad_border.cols());
  for (FMCA::Index i = 0; i < P_quad_border.cols(); ++i) {
    u_bc[i] = 0;
  }
  // Right hand side
  FMCA::Vector f(P_quad_interior.cols());
  for (FMCA::Index i = 0; i < P_quad_interior.cols(); ++i) {
    f[i] = 6;
  }
  std::cout << "u_bc and f created" << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 1. / 3;
  const FMCA::Index dtilde = 4;
  const FMCA::Scalar threshold_kernel = 0;
  // 1e0 / NPTS_QUAD_INTERIOR;
  const FMCA::Scalar threshold_gradKernel = 0;
  // 1e1 / NPTS_QUAD_INTERIOR;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 2 * (dtilde - 1);
  const FMCA::Scalar beta = 10000;
  const std::string kernel_type = "MATERN32";

  const Moments mom_sources(P_sources, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  FMCA::Vector minDistance = minDistanceVector(hst_sources, P_sources);
  FMCA::Scalar sigma_h = minDistance.mean();
  std::cout << "average distance:                      " << sigma_h << std::endl;

  std::vector<double> sigmas = {1.5};

  for (FMCA::Scalar sigma_factor : sigmas) {
    FMCA::Scalar sigma = sigma_factor * sigma_h;

    FMCA::Vector u = SolvePoisson_MonteCarlo(
        DIM, P_sources, P_quad_interior, w, P_quad_border, w_border, Normals,
        u_bc, f, sigma, eta, dtilde, threshold_kernel, threshold_gradKernel,
        threshold_weights, MPOLE_DEG, beta, kernel_type);

    FMCA::Vector u_grid = hst_sources.inverseSampletTransform(u);
    u_grid = hst_sources.toNaturalOrder(u_grid);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Cumpute the solution K*u with the compression of the kernel matrix K
    const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);

    // plot solution
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
      double z = Peval(2, i);
      analytical_sol[i] = 1 - x*x - y*y - z*z;
    }

    FMCA::Scalar error = (rec - analytical_sol).norm() / analytical_sol.norm();
    std::cout << "Error:                                            " << error
              << std::endl;

    //   FMCA::Vector absolute_error(Peval.cols());
    //   for (int i = 0; i < Peval.cols(); ++i) {
    //     absolute_error[i] = abs(rec[i] - analytical_sol[i]);
    //   }

    //   // plot error
    //   {
    //     FMCA::Plotter2D plotter_error;
    //     plotter_error.plotFunction(outputNameErrorPlot, Peval,
    //     absolute_error);
    //   }
    // }
  }

  return 0;
}
