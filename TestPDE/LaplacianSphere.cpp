/* We solve here the Poisson problem on the unit shpere
- laplacian(u) = 6
u = 0 on \partial u
Using Matern32 kernel and penalty method to impose the boundary conditions.
We rely on the FMCA library by M.Multerer.
 */

#include "SolvePoisson.h"
///////////////////////////////////
#include <random>

#include "../FMCA/src/util/Plotter.h"
#include "read_files_txt.h"

#define DIM 3

// Function to generate uniformly distributed points within the unit sphere
FMCA::Matrix generateQuadraturePoints(int M) {
  FMCA::Matrix points(3, M);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  for (int i = 0; i < M; ++i) {
    double u = dis(gen);
    double theta = dis(gen) * 2 * FMCA_PI;
    double phi = std::acos(2 * u - 1);
    double r = std::cbrt(
        dis(gen));  // Cube root to ensure uniform distribution in volume

    points(0, i) = r * std::sin(phi) * std::cos(theta);
    points(1, i) = r * std::sin(phi) * std::sin(theta);
    points(2, i) = r * std::cos(phi);
  }

  return points;
}

// Function to generate uniformly distributed points on the unit sphere
FMCA::Matrix generateQuadraturePointsSurface(int M) {
  FMCA::Matrix points(3, M);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d(0, 1);

  for (int i = 0; i < M; ++i) {
    double x = d(gen);
    double y = d(gen);
    double z = d(gen);
    double norm = std::sqrt(x * x + y * y + z * z);
    points(0, i) = x / norm;
    points(1, i) = y / norm;
    points(2, i) = z / norm;
  }
  return points;
}

int main() {
  //   int NPTS_QUAD = 10000;
  //   FMCA::Matrix P_quad = generateQuadraturePoints(NPTS_QUAD);
  //   int NPTS_BORDER = 5000;
  //   FMCA::Matrix P_quad_border =
  //   generateQuadraturePointsSurface(NPTS_BORDER);

  FMCA::Matrix P_quad_full;
  FMCA::Matrix P_quad_border_full;
  readTXT("data/MC_Interior_Sphere_Collocation.txt", P_quad_full, DIM);
  readTXT("data/MC_Bnd_Sphere_Collocation.txt", P_quad_border_full, DIM);
  int NPTS_QUAD = 40000;
  FMCA::Matrix P_quad = P_quad_full.leftCols(NPTS_QUAD);
  int NPTS_BORDER = 20000;
  FMCA::Matrix P_quad_border = P_quad_border_full.leftCols(NPTS_BORDER);
  FMCA::Matrix Normals = P_quad_border;

  FMCA::IO::plotPoints("P_sphere_quad.vtk", P_quad);
  FMCA::IO::plotPoints("P_sphere_quadBorder.vtk", P_quad_border);
  // DATA
  //////////////////////////////////////////////////////////////////////////////
  FMCA::Matrix P_sources = P_quad_full.rightCols(5000);
  //////////////////////////////////////////////////////////////////////////////
  // U_BC vector
  FMCA::Vector u_bc(P_quad_border.cols());
  for (FMCA::Index i = 0; i < P_quad_border.cols(); ++i) {
    u_bc[i] = 0;
  }
  // Right hand side
  FMCA::Vector f(P_quad.cols());
  for (FMCA::Index i = 0; i < P_quad.cols(); ++i) {
    f[i] = 6;
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 1. / 3;
  const FMCA::Index dtilde = 6;
  const FMCA::Scalar threshold_kernel = 1e-4;
  const FMCA::Scalar threshold_gradKernel = 1e-2;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 2 * (dtilde - 1);
  const FMCA::Scalar beta = 10000;
  const std::string kernel_type = "Gaussian";

  const Moments mom_sources(P_sources, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  FMCA::Vector minDistance = minDistanceVector(hst_sources, P_sources);
  FMCA::Scalar sigma_h = minDistance.mean();
  std::cout << "fill distance:                      " << sigma_h << std::endl;

  std::vector<double> sigmas = {1.5};

  for (FMCA::Scalar sigma_factor : sigmas) {
    FMCA::Scalar sigma = sigma_factor * sigma_h;
    std::cout << "sigma factor =                         " << sigma
              << std::endl;

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
    const auto &triplets = Kcomp_ss.triplets();
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
    Penalty *= 4 * FMCA_PI / NPTS_BORDER;
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
      gradk *= (4.0 * FMCA_PI / 3.0) / NPTS_QUAD;
      // gradk *= 1 / NPTS_QUAD;
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
    Eigen::SparseMatrix<double> Neumann = GradNormal * KCompressed_source_quadborder.transpose();
    Neumann *= 4 * FMCA_PI / NPTS_BORDER;

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
    // FCompressed *= 1 / NPTS_QUAD;
    FCompressed *= (4.0 * FMCA_PI / 3.0) / NPTS_QUAD;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // GCompressed = right hand side penalty
    FMCA::Vector GCompressed = KCompressed_source_quadborder *
                               hst_quad_border.sampletTransform(u_bc_reordered);
    GCompressed *= 4 * FMCA_PI / NPTS_BORDER;
    // GCompressed *= 1 / NPTS_BORDER;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NCompressed = right hand side penalty
    FMCA::Vector NCompressed = GradNormal * hst_quad_border.sampletTransform(u_bc_reordered);
    NCompressed *= 4 * FMCA_PI / NPTS_BORDER;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    Eigen::SparseMatrix<double> Matrix_system_Upper =
        (Stiffness + beta * Penalty - (Neumann + Neumann_Nitsche));
    std::cout << "Number of entries before ApplyPattern  = "
              << Matrix_system_Upper.nonZeros() / P_sources.cols() << std::endl;
    applyPattern(Matrix_system_Upper, a_priori_triplets);
    std::cout << "Number of entries after ApplyPattern  = "
              << Matrix_system_Upper.nonZeros() / P_sources.cols() << std::endl;

    Eigen::SparseMatrix<double> Matrix_system =
        Matrix_system_Upper.selfadjointView<Eigen::Upper>();
    std::cout << "Number of entries after ApplyPattern  = "
              << Matrix_system.nonZeros() / P_sources.cols() << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Solver
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>
        choleskySolver;
    choleskySolver.compute(Matrix_system);
    // u = solution in Samplet Basis
    FMCA::Vector u = choleskySolver.solve(FCompressed + beta * GCompressed - NCompressed); 
    std::cout << "residual error:                                  "
              << ((Matrix_system)*u - (FCompressed + beta * GCompressed - NCompressed)).norm()
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

    // Analytical sol of the problem
    FMCA::Vector analytical_sol(P_sources.cols());
    for (int i = 0; i < P_sources.cols(); ++i) {
      double x = P_sources(0, i);
      double y = P_sources(1, i);
      double z = P_sources(2, i);
      analytical_sol[i] = 1 - x * x - y * y - z * z;
    }

    FMCA::Scalar error = (rec - analytical_sol).norm() / analytical_sol.norm();
    std::cout << "Error:                                            " << error
              << std::endl;
    std::cout << "--------------------------------------------------"
              << std::endl;
  }

  return 0;
}
