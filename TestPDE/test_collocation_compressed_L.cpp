/* We solve here the Poisson problem on an L shape domain
-laplacian(u) = 1
u = 0 on \partial u
Using Gaussian and Matern52 kernel and Kansa method collocation method.
The matrices are comrpessed using Samplets.
We rely on the FMCA library by M.Multerer.
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
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
using MatrixEvaluatorKernel =
    FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluatorKernel =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

// Analytical soltuion
FMCA::Scalar u(FMCA::Scalar x, FMCA::Scalar y) {
  FMCA::Scalar sum = 0.0;
  // Sum the series for the first 50 odd terms
  for (FMCA::Index k = 1; k <= 199; k += 2) {
    FMCA::Scalar term =
        (sin(k * FMCA_PI * (1 + x) / 2) / (pow(k, 3) * sinh(k * FMCA_PI))) *
        (sinh(k * FMCA_PI * (1 + y) / 2) + sinh(k * FMCA_PI * (1 - y) / 2));
    sum += term;
  }
  return (1 - pow(x, 2)) / 2 - (16 / (FMCA_PI * FMCA_PI * FMCA_PI)) * sum;
}

int main() {
  // Initialize the matrices of source points, quadrature points, weights
  FMCA::Tictoc T;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_sources_border;
  FMCA::Matrix P;
  //////////////////////////////////////////////////////////////////////////////
  readTXT("data/vertices_interior_L.txt", P_sources, 2);
  readTXT("data/vertices_boundary_L.txt", P_sources_border, 2);
  readTXT("data/Int_and_Bnd_L.txt", P, 2);

  int NPTS_SOURCE = P_sources.cols();
  int NPTS_SOURCE_BORDER = P_sources_border.cols();
  int N = P.cols();

  // U_BC vector
  FMCA::Vector u_bc(NPTS_SOURCE_BORDER);
  for (int i = 0; i < NPTS_SOURCE_BORDER; ++i) {
    // double x = P_sources_border(0, i);
    // double y = P_sources_border(1, i);
    u_bc[i] = 0;
  }
  // Right hand side f
  FMCA::Vector f(NPTS_SOURCE);
  for (int i = 0; i < NPTS_SOURCE; ++i) {
    double x = P_sources(0, i);
    double y = P_sources(1, i);
    f[i] = -1;
    // f[i] = -2 * FMCA_PI * FMCA_PI * sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }
  // Analytical solution
  FMCA::Vector analytical_sol(P.cols());
  for (FMCA::Index i = 0; i < P.cols(); ++i) {
    FMCA::Scalar x = P(0, i);
    FMCA::Scalar y = P(1, i);
    analytical_sol[i] = u(x, y);
  }
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar threshold_kernel = 1e-4;
  const FMCA::Scalar MPOLE_DEG = 2 * (dtilde - 1);
  std::vector<double> sigma_values = {0.01};
  std::string kernel_type = "MATERN52";
  std::string kernel_type_laplacian = "MATERN52_SECOND_DERIVATIVE";
  std::string spascii = "results_matlab/collocationL25k";
  std::string filename = "collocationL25k";
  // Output the filename to a text file
  std::ofstream outfile("filename.txt");
  if (outfile.is_open()) {
    outfile << filename << std::endl;
    outfile.close();
  }
  //////////////////////////////////////////////////////////////////////////////
  for (double sigma : sigma_values) {
    // FMCA::Scalar threshold_laplacianKernel = threshold_kernel / (sigma * sigma);
    FMCA::Scalar threshold_laplacianKernel = 1e-2;

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

    // FMCA::H2MatrixBase<FMCA::H2Matrix> A;
    // A.computePattern(hst,hst,eta);
    Eigen::SparseMatrix<double> A(N, N);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    A.makeCompressed();

    std::cout << "A created" << std::endl;

    std::cout << "anz A:                                 "
              << tripletList.size() / N << std::endl;

    //////////////////////////////////////////////////////////////////////////////
    // std::ofstream file("matrixL25k.mtx");
    // file << "%%MatrixMarket matrix coordinate real general\n";
    // file << A.rows() << " " << A.cols() << " " << A.nonZeros() << "\n";

    // for (int k = 0; k < A.outerSize(); ++k)
    //   for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
    //     file << it.row() + 1 << " " << it.col() + 1 << " " << it.value()
    //          << "\n";

    // file.close();

    FMCA::IO::print2spascii(spascii, A, "w");
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

    // Eigen::GMRES<Eigen::SparseMatrix<double>,
    //              Eigen::DiagonalPreconditioner<double>> gmres;
    // gmres.compute(A);
    // Eigen::VectorXd alpha = gmres.solve(b);

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::MetisOrdering<int>>
        Solver;
    Solver.analyzePattern(A);
    Solver.factorize(A);
    if (Solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomposition failed");
    }
    FMCA::Vector alpha = Solver.solve(b);

    FMCA::Vector alpha_natural = hst.toNaturalOrder(alpha);
    FMCA::IO::print2m(
        "results_matlab/solution_collocation_compressed_SampletBasis_L.m",
        "sol_collocation_compressed_SampletBasis_L", alpha, "w");

    // Check if the solution is valid
    std::cout << "residual error:                          "
              << (A * alpha - b).norm() << std::endl;
    // Check the error
    std::cout << "error:                          "
              << (alpha-analytical_sol).norm()/analytical_sol.norm() << std::endl;
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
    Kalpha = hst.inverseSampletTransform(Kalpha);
    Kalpha = Kalpha(inversePermutationVector(hst));

    FMCA::IO::print2m("results_matlab/solution_collocation_compressed_L.m",
                      "sol_collocation_compressed_L", Kalpha, "w");

    FMCA::Vector absolute_error(P_sources.cols());
    for (int i = 0; i < P_sources.cols(); ++i) {
      absolute_error[i] = abs(Kalpha[i] - analytical_sol[i]);
    }

    FMCA::IO::print2m(
        "results_matlab/error_solution_collocation_compressed_L.m",
        "absolute_error_solution_collocation_compressed_L", absolute_error, "w");
  }
  return 0;
}
