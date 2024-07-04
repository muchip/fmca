#include <algorithm>
#include <cstdlib>
#include <iostream>
// ##############################
#include <Eigen/Dense>
#include <Eigen/MetisSupport>

#include "../FMCA/GradKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/Samplets/adaptiveTreeSearch.h"
#include "../FMCA/src/util/Grid3D.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "FunctionsPDE.h"

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

FMCA::Vector SolvePoisson_not_compressed(
    const FMCA::Scalar &DIM, FMCA::Matrix &P_sources, FMCA::Matrix &P_quad,
    FMCA::Vector &w_vec, FMCA::Matrix &P_quad_border,
    FMCA::Vector &w_vec_border, FMCA::Matrix &Normals, FMCA::Vector &u_bc,
    FMCA::Vector &f, const FMCA::Scalar &sigma, const FMCA::Scalar &beta,
    const std::string &kernel_type) {
  FMCA::Tictoc T;

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Number of source points =                         "
            << P_sources.cols() << std::endl;
  std::cout << "Number of quad points =                           "
            << P_quad.cols() << std::endl;
  std::cout << "Number of quad points border =                    "
            << P_quad_border.cols() << std::endl;
  std::cout << "sigma =                                           " << sigma
            << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Kernel source-source compression
  const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);

  FMCA::Matrix K_ss = kernel_funtion_ss.eval(P_sources, P_sources);
  std::cout << "K_ss computed                         " << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Kernel (sources,quad) compression
  const FMCA::CovarianceKernel kernel_funtion(kernel_type, sigma);
  FMCA::Matrix K_sq = kernel_funtion.eval(P_sources, P_quad);
  std::cout << "K_sq computed                         " << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Penalty: K_{Psources, Pquad_border} * W_border
  // * K_{Psources, Pquad_border}.transpose()
  const FMCA::CovarianceKernel kernel_funtion_sources_quadborder(kernel_type,
                                                                 sigma);
  FMCA::Matrix K_sqb = kernel_funtion_sources_quadborder.eval(P_sources, P_quad_border);
  FMCA::Matrix Penalty = K_sqb * K_sqb.transpose();
  Penalty *= w_vec_border(0);
  std::cout << "Penalty computed                         " << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // create Stiffness and Neumann term
  FMCA::Matrix Stiffness = FMCA::Matrix::Zero(P_sources.cols(), P_sources.cols());

  FMCA::Matrix GradNormal = FMCA::Matrix::Zero(P_sources.cols(), P_quad_border.cols());

  for (FMCA::Index i = 0; i < DIM; ++i) {
    const FMCA::GradKernel function(kernel_type, sigma, 1, i);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Stiffness
    FMCA::Matrix grad = function.eval(P_sources, P_quad);

    FMCA::Matrix stiff = grad * grad.transpose();
    stiff *= w_vec(0);
    Stiffness += stiff;
    std::cout << "Grad computed                         " << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Neumann
    FMCA::Matrix gradK_neumann = function.eval(P_sources, P_quad_border);

    FMCA::Matrix gradk_n = gradK_neumann * Normals.row(i).asDiagonal();
    GradNormal += gradk_n;
    std::cout << "GradNormal                         " << std::endl;
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Final Neumann Term
  FMCA::Matrix Neumann = GradNormal * K_sqb.transpose();
  Neumann *= w_vec_border(0);
  // Nitsche’s Term
  FMCA::Matrix Neumann_Nitsche = Neumann.transpose();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // FCompressed = right hand side of the problem involving the source term f
  FMCA::Vector FCompressed = K_sq * f;
  FCompressed *= w_vec(0);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // GCompressed = right hand side penalty
  FMCA::Vector GCompressed = K_sqb * u_bc;
  GCompressed *= w_vec_border(0);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // NCompressed= right hand side Nitsche
  FMCA::Vector NCompressed = GradNormal * u_bc;
  NCompressed *= w_vec_border(0);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  FMCA::Matrix Matrix_system =
      (Stiffness + beta * Penalty - (Neumann + Neumann_Nitsche));

  FMCA::Vector Matrix_system_diagonal = Matrix_system.diagonal();
  std::cout << "Min element diagonal:                               "
            << Matrix_system_diagonal.minCoeff() << std::endl;

  std::cout << "Number of element per row system matrix:            "
            << Matrix_system.nonZeros() / Matrix_system.rows() << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Solver
  Eigen::ColPivHouseholderQR<FMCA::Matrix> solver;
  solver.compute(Matrix_system);
  // u = solution in Samplet Basis
  FMCA::Vector u = solver.solve(FCompressed + beta * GCompressed - NCompressed);

  std::cout << "residual error:                                  "
            << ((Matrix_system)*u -
                (FCompressed + beta * GCompressed - NCompressed))
                   .norm()
            << std::endl;
  return u;
}


#define DIM 3

FMCA::Vector RandomPointInterior() {
  FMCA::Vector point_interior;
  do {
    point_interior = Eigen::Vector3d::Random();
  } while (point_interior.squaredNorm() >= 1.0);
  return point_interior;
}

FMCA::Vector RandomPointBoundary() {
  FMCA::Vector point_bnd = RandomPointInterior();
  return point_bnd / point_bnd.norm();
}

FMCA::Matrix MonteCarloPointsInterior(int N) {
  FMCA::Matrix P_interior(DIM, N);
  for (int i = 0; i < N; ++i) {
    FMCA::Vector x = RandomPointInterior();
    P_interior.col(i) = x;
  }
  return P_interior;
}

FMCA::Matrix MonteCarloPointsBoundary(int N) {
  FMCA::Matrix P_bnd(DIM, N);
  for (int i = 0; i < N; ++i) {
    FMCA::Vector x = RandomPointBoundary();
    P_bnd.col(i) = x;
  }
  return P_bnd;
}

FMCA::Scalar WeightInterior(int N, FMCA::Scalar r) {
  return (4. / 3.) * FMCA_PI * r * r * r *
         (1. / N);  // Monte Carlo integration weights: Volume*1/N
}

FMCA::Scalar WeightBnd(int N, FMCA::Scalar r) {
  return 4. * FMCA_PI * r * r *
         (1. / N);  // Monte Carlo integration weights: Surface*1/N
}

int main() {
  // DATA
  FMCA::Tictoc T;

  int N_interior = 100;
  FMCA::Matrix P_interior = MonteCarloPointsInterior(N_interior);
  FMCA::IO::plotPoints("P_interior.vtk", P_interior);

  int N_bnd = 20;
  FMCA::Matrix P_bnd = MonteCarloPointsBoundary(N_bnd);
  FMCA::IO::plotPoints("P_boundary.vtk", P_bnd);


  FMCA::Matrix Normals = P_bnd;

  FMCA::Scalar factor = 0.2;
  int N_interior_sources = factor * N_interior;
  int N_bnd_sources = factor * N_bnd;
  FMCA::Matrix P_interior_sources =
      MonteCarloPointsInterior(N_interior_sources);
  FMCA::Matrix P_bnd_sources = MonteCarloPointsBoundary(N_bnd_sources);

  FMCA::Matrix P(DIM, N_interior_sources + N_bnd_sources);
  P << P_interior_sources, P_bnd_sources;

  //   FMCA::IO::plotPoints("P_sphere.vtk", P);
  std::cout << P.transpose() << std::endl; 

  FMCA::Scalar w = WeightInterior(N_interior, 1.0);
  std::cout << "w_vec:                      " << w << std::endl;
  FMCA::Vector w_vec = Eigen::VectorXd::Ones(N_interior);
  w_vec *= w;

  FMCA::Scalar w_border = WeightBnd(N_bnd, 1.0);
  std::cout << "w_vec_border:               " << w_border << std::endl;
  FMCA::Vector w_vec_border = Eigen::VectorXd::Ones(N_bnd);
  w_vec_border *= w_border;
  // U_BC vector
  FMCA::Vector u_bc(P_bnd.cols());
  for (FMCA::Index i = 0; i < P_bnd.cols(); ++i) {
    u_bc[i] = 0;
  }
  // Right hand side
  FMCA::Vector f(P_interior.cols());
  for (FMCA::Index i = 0; i < P_interior.cols(); ++i) {
    f[i] = 6;
  }
  // Analytical sol of the problem
  FMCA::Vector analytical_sol(P.cols());
  for (int i = 0; i < P.cols(); ++i) {
    FMCA::Scalar x = P(0, i);
    FMCA::Scalar y = P(1, i);
    FMCA::Scalar z = P(2, i);
    analytical_sol[i] = 1 - x * x - y * y - z * z;
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar MPOLE_DEG = 4;
  const FMCA::Scalar beta = 1000;
  const std::string kernel_type = "MATERN32";

  const Moments mom_sources(P, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P);
  FMCA::Vector minDistance = minDistanceVector(hst_sources, P);

  // fill distance
  auto maxElementIterator =
      std::max_element(minDistance.begin(), minDistance.end());
  FMCA::Scalar sigma_h = *maxElementIterator;
  std::cout << "fill distance:                      " << sigma_h << std::endl;

  std::vector<double> sigmas = {0.8};

  for (FMCA::Scalar sigma_factor : sigmas) {
    std::cout << "sigma factor =                         " << sigma_factor
              << std::endl;
    FMCA::Scalar sigma = sigma_factor * sigma_h;

    FMCA::Vector u = SolvePoisson_not_compressed(
        DIM, P, P_interior, w_vec, P_bnd, w_vec_border, Normals, u_bc, f, sigma,
        beta, kernel_type);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Cumpute the solution K*u with the compression of the kernel matrix K
    const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
    FMCA::Matrix K = kernel_funtion_ss.eval(P, P);

    FMCA::Vector KU = K * u;
    std::cout<< "Ku" << std::endl;
    std::cout<< KU << std::endl;

    std::cout<< "analytical sol" << std::endl;
    std::cout<< analytical_sol << std::endl;

    // Error
    FMCA::Scalar error = (KU - analytical_sol).norm() / analytical_sol.norm();

    std::cout << "Error:                                            " << error
              << std::endl;
  }

  return 0;
}
