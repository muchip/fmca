/* We solve here the Poisson problem on an L shape domain
-laplacian(u) = 1
u = 0 on \partial u
Using Gaussian and Matern52 kernel and Kansa method collocation method.
The matrices are comrpessed using Samplets.
We rely on the FMCA library by M.Multerer.
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
// #include
// </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/MetisSupport>
#include </opt/homebrew/Cellar/metis/5.1.0/include/metis.h>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>
#include <unsupported/Eigen/IterativeSolvers>

#include "../FMCA/GradKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/LaplacianKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/Grid2D.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Plotter.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "read_files_txt.h"

#define DIM 2

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
  FMCA::Matrix P_sources;
  FMCA::Matrix P_sources_border;
  FMCA::Matrix P;
  FMCA::Matrix Peval;
  //////////////////////////////////////////////////////////////////////////////
  // 25k 0.1
  std::vector<double> sigma_values = {0.2};
  readTXT("data/interior_square_6k.txt", P_sources, 2);
  readTXT("data/boundary_L_6k.txt", P_sources_border, 2);
  readTXT("data/int_and_bnd_L_6k.txt", P, 2);
  readTXT("data/InteriorAndBnd_L10k.txt", Peval, 2);

  std::string matrixFilename = "CollocationPatternL6k";
  std::string solName = "CollocationSolutionL6k.vtk";

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
    double x = P_sources(0, i);
    double y = P_sources(1, i);
    f[i] = -10;
  }
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar threshold_kernel = 1e-4;
  const FMCA::Scalar MPOLE_DEG = 2 * (dtilde - 1);
  std::string kernel_type = "MATERN52";
  std::string kernel_type_laplacian = "MATERN52_SECOND_DERIVATIVE";
  //////////////////////////////////////////////////////////////////////////////
  for (double sigma : sigma_values) {
    FMCA::Scalar threshold_laplacianKernel = threshold_kernel / (sigma * sigma);

    std::cout << "Running with sigma = " << sigma << std::endl;
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
    auto maxElementIterator =
        std::max_element(minDistance.begin(), minDistance.end());
    FMCA::Scalar sigma_h = *maxElementIterator;
    std::cout << "fill distance:                      " << sigma_h << std::endl;
    //////////////////////////////////////////////////////////////////////////////
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Number of interior points = " << NPTS_SOURCE << std::endl;
    std::cout << "Number of border points = " << NPTS_SOURCE_BORDER
              << std::endl;
    std::cout << "Total number of points = " << N << std::endl;
    std::cout << "sigma = " << sigma << std::endl;
    //////////////////////////////////////////////////////////////////////////////
    const FMCA::CovarianceKernel kernel_funtion_border(kernel_type, sigma);
    const usMatrixEvaluatorKernel mat_eval_border(mom_border, mom,
                                                  kernel_funtion_border);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> K_border;
    K_border.init(hst_sources_border, hst, eta, threshold_kernel);
    T.tic();
    K_border.compress(mat_eval_border);
    const auto& trips_border = K_border.triplets();
    std::cout << "anz kernel:                                         "
              << trips_border.size() / NPTS_SOURCE_BORDER << std::endl;
    Eigen::SparseMatrix<double> Kcomp_border(NPTS_SOURCE_BORDER, N);
    Kcomp_border.setFromTriplets(trips_border.begin(), trips_border.end());
    Kcomp_border.makeCompressed();
    //////////////////////////////////////////////////////////////////////////////
    Eigen::SparseMatrix<double> Laplacian(NPTS_SOURCE, N);
    Laplacian.setZero();

    for (int i = 0; i < DIM; ++i) {
      const FMCA::GradKernel laplacian_funtion(kernel_type_laplacian, sigma, 1,
                                               i);
      const usMatrixEvaluator mat_eval_laplacian(mom_sources, mom,
                                                 laplacian_funtion);
      // gradKernel compression
      FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
          Laplacian_Interior;
      Laplacian_Interior.init(hst_sources, hst, eta, threshold_laplacianKernel);
      T.tic();
      Laplacian_Interior.compress(mat_eval_laplacian);
      const auto& trips_Laplacian = Laplacian_Interior.triplets();
      std::cout << "anz laplacianKernel:                                 "
                << trips_Laplacian.size() / NPTS_SOURCE << std::endl;
      Eigen::SparseMatrix<double> Lcomp_Interior(NPTS_SOURCE, N);
      Lcomp_Interior.setFromTriplets(trips_Laplacian.begin(),
                                     trips_Laplacian.end());
      Lcomp_Interior.makeCompressed();
      //////////////////////////////////////////////////////////////////////////////
      Laplacian += Lcomp_Interior;
    }
    std::cout << "Kernel and Laplacian created" << std::endl;

    // Concatenate Laplacian matrix and Kernel_border vertically
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(Laplacian.nonZeros() + Kcomp_border.nonZeros());
    // Add non-zeros of Laplacian
    for (int k = 0; k < Laplacian.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Laplacian, k); it;
           ++it) {
        tripletList.push_back(
            Eigen::Triplet<double>(it.row(), it.col(), it.value()));
      }
    }
    // Add non-zeros of Kcomp_border
    for (int k = 0; k < Kcomp_border.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Kcomp_border, k); it;
           ++it) {
        tripletList.push_back(Eigen::Triplet<double>(
            it.row() + Laplacian.rows(), it.col(), it.value()));
      }
    }

    Eigen::SparseMatrix<double> A(N, N);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    A.makeCompressed();

    std::cout << "A created" << std::endl;

    std::cout << "anz A:                                 "
              << tripletList.size() / N << std::endl;
    FMCA::IO::print2spascii(matrixFilename, A, "w");
    //////////////////////////////////////////////////////////////////////////////

    FMCA::Vector f_reordered = f(permutationVector(hst_sources));
    FMCA::Vector f_transformed = hst_sources.sampletTransform(f_reordered);

    FMCA::Vector u_bc_reordered = u_bc(permutationVector(hst_sources_border));
    FMCA::Vector u_bc_transformed =
        hst_sources_border.sampletTransform(u_bc_reordered);
    // Concatenate f vector and u_bc vector vertically
    Eigen::VectorXd b(N);
    b << f_transformed, u_bc_transformed;

    std::cout << "RHS created" << std::endl;
    std::string solver_type = "qr";
    std::cout << "Solver                               " << solver_type
              << std::endl;

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::MetisOrdering<int>>
        Solver;
    Solver.analyzePattern(A);
    Solver.factorize(A);
    if (Solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomposition failed");
    }
    FMCA::Vector alpha = Solver.solve(b);

    Eigen::Index maxIndex, minIndex;
    double maxValue = alpha.maxCoeff(&maxIndex);
    double minValue = alpha.minCoeff(&minIndex);
    std::cout << "highest samplet coefficient: " << maxValue << " at index "
              << maxIndex << std::endl;
    std::cout << "lowest samplet coefficient: " << minValue << " at index "
              << minIndex << std::endl;

    FMCA::Vector alpha_grid = hst.inverseSampletTransform(alpha);
    alpha_grid = hst.toNaturalOrder(alpha_grid);
    Eigen::Index maxIndex_NO, minIndex_NO;
    double maxValue_NO = alpha_grid.maxCoeff(&maxIndex_NO);
    double minValue_NO = alpha_grid.minCoeff(&minIndex_NO);
    std::cout << "highest samplet coefficient natural order: " << maxValue_NO
              << " at index " << maxIndex_NO << " corresponding to point ["
              << P.col(maxIndex_NO)[0] << " " << P.col(maxIndex_NO)[1] << "]"
              << std::endl;
    std::cout << "lowest samplet coefficient natural order: " << minValue_NO
              << " at index " << minIndex_NO << " corresponding to point ["
              << P.col(minIndex_NO)[0] << " " << P.col(minIndex_NO)[1] << "]"
              << std::endl;

    // Check if the solution is valid
    std::cout << "residual error:                          "
              << (A * alpha - b).norm() << std::endl;

    //////////////////////////////////////////////////////////////////////////////
    const FMCA::CovarianceKernel kernel_funtion_N(kernel_type, sigma);
    const MatrixEvaluatorKernel mat_eval_kernel_N(mom, kernel_funtion_N);
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_N;
    Kcomp_N.init(hst, eta, threshold_kernel);
    Kcomp_N.compress(mat_eval_kernel_N);
    const auto& trips_N = Kcomp_N.triplets();
    std::cout << "anz kernel ss:                                      "
              << trips_N.size() / N << std::endl;
    Eigen::SparseMatrix<double> Kcomp_N_Sparse(N, N);
    Kcomp_N_Sparse.setFromTriplets(trips_N.begin(), trips_N.end());
    Kcomp_N_Sparse.makeCompressed();

    FMCA::Vector Kalpha =
        Kcomp_N_Sparse.selfadjointView<Eigen::Upper>() * alpha;
    FMCA::Vector Kalpha_transformed = hst.inverseSampletTransform(Kalpha);
    FMCA::Vector Kalpha_permuted =
        Kalpha_transformed(inversePermutationVector(hst));

    // plot solution
    {
      const Moments cmom(P, MPOLE_DEG);
      const Moments rmom(Peval, MPOLE_DEG);
      const H2ClusterTree hct(cmom, 0, P);
      const H2ClusterTree hct_eval(rmom, 0, Peval);
      const FMCA::Vector sx = hct.toClusterOrder(alpha_grid);
      const usMatrixEvaluatorKernel mat_eval(rmom, cmom, kernel_funtion_N);
      FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster> hmat;
      hmat.computePattern(hct_eval, hct, eta);
      hmat.statistics();
      FMCA::Vector srec = hmat.action(mat_eval, sx);
      FMCA::Vector rec = hct_eval.toNaturalOrder(srec);
      FMCA::Plotter2D plotter;
      plotter.plotFunction(solName, Peval, rec);
    }
  }
  return 0;
}