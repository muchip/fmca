/* We solve here the Poisson problem on the unit circle
- laplacian(u) = 4
u = 0 on \partial u
Using Matern32 kernel and penalty method to impose the boundary conditions.
We rely on the FMCA library by M.Multerer.
 */

#include <algorithm>
#include <cstdlib>
#include <iostream>
// ##############################
#include <Eigen/Eigenvalues>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

#include "../FMCA/GradKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/Samplets/adaptiveTreeSearch.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "FunctionsPDE.h"
///////////////////////////////////
#include "../FMCA/src/util/Plotter.h"
#include "read_files_txt.h"

#define DIM 2

using EigenCholesky =
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>;

FMCA::Vector RandomPointInterior() {
  FMCA::Vector point_interior;
  do {
    point_interior = Eigen::Vector2d::Random();
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
  return FMCA_PI * r * r *
         (1. / N);  // Monte Carlo integration weights: Volume*1/N
}

FMCA::Scalar WeightBnd(int N, FMCA::Scalar r) {
  return 2. * FMCA_PI * r *
         (1. / N);  // Monte Carlo integration weights: Surface*1/N
}


void saveMatrixToFile(const FMCA::Matrix& matrix, const std::string& filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    for (int i = 0; i < matrix.rows(); ++i) {
      for (int j = 0; j < matrix.cols(); ++j) {
        file << matrix(i, j) << " ";
      }
      file << "\n";
    }
    std::cout << "Matrix saved to " << filename << std::endl;
    file.close();
  } else {
    std::cerr << "Unable to open file " << filename << std::endl;
  }
}

int main() {
  // DATA
  FMCA::Tictoc T;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_interior_full;
  ;
  FMCA::Matrix P_bnd_full;

  // int N_interior = 1000000;
  // FMCA::Matrix P_interior = MonteCarloPointsInterior(N_interior);
  // saveMatrixToFile(P_interior.transpose(), "MC_Interior_Circle.txt");
  // int N_bnd = 500000;
  // FMCA::Matrix P_bnd = MonteCarloPointsBoundary(N_bnd);
  // saveMatrixToFile(P_bnd.transpose(), "MC_Bnd_Circle.txt");

  readTXT("data/MC_Interior_Circle.txt", P_interior_full, DIM);
  readTXT("data/MC_Bnd_Circle.txt", P_bnd_full, DIM);

  int N_interior = 100000;
  int N_bnd = 50000;
  FMCA::Matrix P_quad = P_interior_full.leftCols(N_interior);
  FMCA::Matrix P_quad_border = P_bnd_full.leftCols(N_bnd);
  FMCA::IO::plotPoints2D("P_circle_interior.vtk", P_quad);
  FMCA::IO::plotPoints2D("P_circle_bnd.vtk", P_quad_border);

  FMCA::Matrix Normals = P_quad_border;

  readTXT("data/Points_Circle_Uniform.txt", P_sources, DIM);
  FMCA::IO::plotPoints2D("P_circle_uniform.vtk", P_sources);
  std::cout << "P done" << std::endl;

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // U_BC vector
  FMCA::Vector u_bc(P_quad_border.cols());
  for (FMCA::Index i = 0; i < P_quad_border.cols(); ++i) {
    u_bc[i] = 0;
  }
  // Right hand side
  FMCA::Vector f(P_quad.cols());
  for (FMCA::Index i = 0; i < P_quad.cols(); ++i) {
    f[i] = 4;
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 1. / DIM;
  const FMCA::Index dtilde = 6;
  const FMCA::Scalar threshold_kernel = 1e-6;
  const FMCA::Scalar threshold_gradKernel = 1e-3;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 2 * (dtilde - 1);
  const FMCA::Scalar beta = 10000;
  const std::string kernel_type = "MATERN32";

  const Moments mom_sources(P_sources, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  FMCA::Vector minDistance = minDistanceVector(hst_sources, P_sources);
  FMCA::Scalar sigma_h = minDistance.mean();
  std::cout << "average distance:                   " << sigma_h << std::endl;

  std::cout << "Number of Interior Points:          " << N_interior
            << std::endl;
  std::cout << "Number of Bnd Points:               " << N_bnd << std::endl;
  std::cout << "Number of Source Points:            " << P_sources.cols()
            << std::endl;

  std::vector<double> sigmas = {1,2,3};
  for (FMCA::Scalar sigma_factor : sigmas) {
    FMCA::Scalar sigma = sigma_factor * sigma_h;
    std::cout << "sigma =                             " << sigma << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    const Moments mom_sources(P_sources, MPOLE_DEG);
    const Moments mom_quad(P_quad, MPOLE_DEG);
    const Moments mom_quad_border(P_quad_border, MPOLE_DEG);
    const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
    const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
    const SampletMoments samp_mom_quad_border(P_quad_border, dtilde - 1);
    H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
    H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
    H2SampletTree hst_quad_border(mom_quad_border, samp_mom_quad_border, 0,
                                  P_quad_border);

    // Kernel source-source compression
    const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
    const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources,
                                                   kernel_funtion_ss);
    std::cout << "Kernel Source-Source" << std::endl;
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
    Kcomp_ss.init(hst_sources, eta, threshold_kernel);
    std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets =
        Kcomp_ss.a_priori_pattern_triplets();
    Kcomp_ss.compress(mat_eval_kernel_ss);
    const auto& triplets = Kcomp_ss.triplets();
    std::cout << "Size a priori triplets:  "
              << a_priori_triplets.size() / P_sources.cols() << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Penalty: K_{Psources, Pquad_border} * W_border
    // * K_{Psources, Pquad_border}.transpose()
    const FMCA::CovarianceKernel kernel_funtion_sources_quadborder(kernel_type,
                                                                   sigma);
    const usMatrixEvaluatorKernel mat_eval_kernel_sources_quadborder(
        mom_sources, mom_quad_border, kernel_funtion_sources_quadborder);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
        Kcomp_sources_quadborder;
    std::cout << "Kernel Source-QuadBorder" << std::endl;
    Eigen::SparseMatrix<double> KCompressed_source_quadborder =
        createCompressedSparseMatrixUnSymmetric(
            kernel_funtion_sources_quadborder,
            mat_eval_kernel_sources_quadborder, hst_sources, hst_quad_border,
            eta, threshold_kernel, P_sources, P_quad_border);

    Eigen::SparseMatrix<double> Penalty =
        KCompressed_source_quadborder *
        KCompressed_source_quadborder.transpose();
    Penalty *= 2 * FMCA_PI / N_bnd;
    // Penalty *= 1 / NPTS_BORDER;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // create Stiffness and Neumann term
    Eigen::SparseMatrix<double> Stiffness(P_sources.cols(), P_sources.cols());
    Stiffness.setZero();

    Eigen::SparseMatrix<double> GradNormal(P_sources.cols(),
                                           P_quad_border.cols());
    GradNormal.setZero();

    for (FMCA::Index i = 0; i < DIM; ++i) {
      const FMCA::GradKernel function(kernel_type, sigma, 1, i);
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /// Stiffness
      const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
      FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
      std::cout << "GradK" << std::endl;
      Eigen::SparseMatrix<double> GradKCompressed =
          createCompressedSparseMatrixUnSymmetric(
              function, mat_eval, hst_sources, hst_quad, eta,
              threshold_gradKernel, P_sources, P_quad);

      Eigen::SparseMatrix<double> gradk =
          GradKCompressed * GradKCompressed.transpose();
      gradk *= FMCA_PI / N_bnd;
      Stiffness += gradk;

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /// Neumann
      const usMatrixEvaluator mat_eval_neumann(mom_sources, mom_quad_border,
                                               function);
      FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
          Scomp_neumann;
      std::cout << "Neumann" << std::endl;
      Eigen::SparseMatrix<double> GradKCompressed_neumann =
          createCompressedSparseMatrixUnSymmetric(
              function, mat_eval_neumann, hst_sources, hst_quad_border, eta,
              threshold_gradKernel, P_sources, P_quad_border);

      Eigen::SparseMatrix<double> gradk_n =
          NeumannNormalMultiplication(GradKCompressed_neumann, Normals.row(i));
      gradk_n.makeCompressed();
      GradNormal += gradk_n;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Final Neumann Term
    Eigen::SparseMatrix<double> Neumann =
        GradNormal * KCompressed_source_quadborder.transpose();
    Neumann *= 2 * FMCA_PI / N_bnd;

    Eigen::SparseMatrix<double> Neumann_Nitsche = Neumann.transpose();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // reorder the boundary conditions and the rhs to follow the samplets order
    FMCA::Vector u_bc_reordered = hst_quad_border.toClusterOrder(u_bc);
    FMCA::Vector f_reordered = hst_quad.toClusterOrder(f);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Kernel (sources,quad) compression
    const FMCA::CovarianceKernel kernel_funtion(kernel_type, sigma);
    const usMatrixEvaluatorKernel mat_eval_kernel(mom_sources, mom_quad,
                                                  kernel_funtion);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
    std::cout << "Kernel Source-Quad" << std::endl;
    Eigen::SparseMatrix<double> KCompressed_sourcequad =
        createCompressedSparseMatrixUnSymmetric(
            kernel_funtion, mat_eval_kernel, hst_sources, hst_quad, eta,
            threshold_kernel, P_sources, P_quad);

    // FCompressed = right hand side of the problem involving the source term f
    FMCA::Vector FCompressed =
        KCompressed_sourcequad * hst_quad.sampletTransform(f_reordered);
    FCompressed *= FMCA_PI / N_interior;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // GCompressed = right hand side penalty
    FMCA::Vector GCompressed = KCompressed_source_quadborder *
                               hst_quad_border.sampletTransform(u_bc_reordered);
    GCompressed *= 2 * FMCA_PI / N_bnd;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NCompressed = right hand side penalty
    FMCA::Vector NCompressed =
        GradNormal * hst_quad_border.sampletTransform(u_bc_reordered);
    NCompressed *= 2 * FMCA_PI / N_bnd;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Eigen::SparseMatrix<double> Matrix_system_Upper =
        (Stiffness + beta * Penalty - (Neumann + Neumann_Nitsche));

    std::cout << "Number of entries before ApplyPattern  = "
              << Matrix_system_Upper.nonZeros() / P_sources.cols() << std::endl;

    ////////////////////////////////////////////////////////

    applyPattern(Matrix_system_Upper, a_priori_triplets);
    std::cout << "Number of entries after ApplyPattern  = "
              << Matrix_system_Upper.nonZeros() / P_sources.cols() << std::endl;

    Eigen::SparseMatrix<double> Matrix_system =
        Matrix_system_Upper.selfadjointView<Eigen::Upper>();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Solver
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>
        choleskySolver;
    choleskySolver.compute(Matrix_system);
    // u = solution in Samplet Basis
    FMCA::Vector u =
        choleskySolver.solve(FCompressed + beta * GCompressed - NCompressed);
    std::cout << "residual error:                                  "
              << ((Matrix_system)*u -
                  (FCompressed + beta * GCompressed - NCompressed))
                     .norm()
              << std::endl;

    FMCA::Vector u_grid = hst_sources.inverseSampletTransform(u);
    u_grid = hst_sources.toNaturalOrder(u_grid);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // plot solution
    const Moments cmom(P_sources, MPOLE_DEG);
    const Moments rmom(P_sources, MPOLE_DEG);
    const H2ClusterTree hct(cmom, 0, P_sources);
    const H2ClusterTree hct_eval(rmom, 0, P_sources);
    const FMCA::Vector sx = hct.toClusterOrder(u_grid);
    const usMatrixEvaluatorKernel mat_eval(rmom, cmom, kernel_funtion_ss);
    FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster> hmat;
    hmat.computePattern(hct_eval, hct, eta);
    hmat.statistics();
    FMCA::Vector srec = hmat.action(mat_eval, sx);
    FMCA::Vector rec = hct_eval.toNaturalOrder(srec);
    // FMCA::Plotter2D plotter;
    // plotter.plotFunction(outputNamePlot, Peval, rec);

    FMCA::Vector analytical_sol(P_sources.cols());
    for (int i = 0; i < P_sources.cols(); ++i) {
      double x = P_sources(0, i);
      double y = P_sources(1, i);
      analytical_sol[i] = 1 - x * x - y * y;
    }

    FMCA::Scalar error = (rec - analytical_sol).norm() / analytical_sol.norm();
    std::cout << "Error:                                            " << error
              << std::endl;
    std::cout << "--------------------------------------------------"
              << std::endl;
  }

  return 0;
}
