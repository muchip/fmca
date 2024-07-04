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
#include <random>
/////////////////////////////////////////////////
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

FMCA::Vector generateNormalVector(int n) {
  FMCA::Vector X(n);
  std::random_device rd;   // Seed for the random number engine
  std::mt19937 gen(rd());  // Standard mersenne_twister_engine
  std::normal_distribution<> d(
      0, 1);  // Normal distribution with mean 0 and stddev 1

  for (int i = 0; i < n; ++i) {
    X[i] = d(gen);
  }

  return X / X.norm();
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
    FMCA::Vector x = generateNormalVector(DIM);
    P_bnd.col(i) = x;
  }
  return P_bnd;
}

// Function to save matrix to a text file
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
    file.close();  // Close the file
  } else {
    std::cerr << "Unable to open file " << filename << std::endl;
  }
}

int main() {
  FMCA::Tictoc T;
  FMCA::Matrix P_interior_full;
  FMCA::Matrix P_bnd_full;
   readTXT("data/MC_Interior_Circle_collocation.txt", P_interior_full, DIM);
  readTXT("data/MC_Bnd_Circle_collocation.txt", P_bnd_full, DIM);
  
  // Parameters
  const int NPTS_INTERIOR = 4000;
  const int NPTS_BORDER = 2000;
  const int eval_interior = 1000;

  const FMCA::Scalar sigma = 2;

  const FMCA::Scalar eta = 1./3.;
  const FMCA::Index dtilde = 6;
  const FMCA::Scalar threshold_kernel = 0;
  const FMCA::Scalar threshold_laplacianKernel = threshold_kernel*sigma*sigma;
  const FMCA::Scalar MPOLE_DEG = 2 * (dtilde - 1);
  const std::string kernel_type = "MATERN52";
  const std::string kernel_type_laplacian = "MATERN52_SECOND_DERIVATIVE";

  // Points
  FMCA::Matrix P_sources = P_interior_full.leftCols(NPTS_INTERIOR);
  FMCA::Matrix P_sources_border = P_bnd_full.leftCols(NPTS_BORDER);

  FMCA::Matrix P(DIM, NPTS_INTERIOR + NPTS_BORDER);
  P << P_sources, P_sources_border;
  FMCA::IO::plotPoints("P_sphere.vtk", P);
  int N = P.cols();

  FMCA::Matrix Peval = P_interior_full.rightCols(eval_interior);

  /////////////////////////////////////////////////////////////////////////////
  // U_BC vector
  FMCA::Vector u_bc(NPTS_BORDER);
  for (int i = 0; i < NPTS_BORDER; ++i) {
    u_bc[i] = 0;
  }
  // Right hand side f
  FMCA::Vector f(NPTS_INTERIOR);
  for (int i = 0; i < NPTS_INTERIOR; ++i) {
    f[i] = -6;
  }
  //////////////////////////////////////////////////////////////////////////////
    std::cout << std::string(80, '-') << std::endl;
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
    std::cout << "Number of interior points           " << NPTS_INTERIOR << std::endl;
    std::cout << "Number of border points             " << NPTS_BORDER << std::endl;
    std::cout << "Total number of points              " << N << std::endl;
    std::cout << "sigma                               " << sigma << std::endl;
    //////////////////////////////////////////////////////////////////////////////
    // compressed
    const FMCA::CovarianceKernel kernel_funtion_border(kernel_type, sigma);
    const usMatrixEvaluatorKernel mat_eval_border(mom_border, mom,
                                                  kernel_funtion_border);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> K_border;
    K_border.init(hst_sources_border, hst, eta, threshold_kernel);
    T.tic();
    K_border.compress(mat_eval_border);
    const auto& trips_border = K_border.triplets();
    std::cout << "anz kernel:                     "
              << trips_border.size() / NPTS_BORDER << std::endl;
    Eigen::SparseMatrix<double> Kcomp_border(NPTS_BORDER, N);
    Kcomp_border.setFromTriplets(trips_border.begin(), trips_border.end());
    Kcomp_border.makeCompressed();



    // full eval
    auto Perm_bnd = FMCA::permutationMatrix(hst_sources_border);
    auto Perm_N = FMCA::permutationMatrix(hst);

    FMCA::Matrix K_bnd_n_eval = kernel_funtion_border.eval(P_sources_border,P);
    FMCA::Matrix K_bnd_n_eval_permuted = Perm_bnd.transpose() * K_bnd_n_eval * Perm_N;
    //////////////////////////////////////////////////////////////////////////////
    Eigen::SparseMatrix<double> Laplacian(NPTS_INTERIOR, N);
    Laplacian.setZero();


    FMCA::Matrix Laplacian_eval(NPTS_INTERIOR, N);
    Laplacian_eval.setZero();
    FMCA::Matrix Laplacian_eval_permuted(NPTS_INTERIOR, N);
    Laplacian_eval_permuted.setZero();

    for (int i = 0; i < DIM; ++i) {
      // comrpessed
      const FMCA::GradKernel laplacian_funtion(kernel_type_laplacian, sigma, 1,
                                               i);
      const usMatrixEvaluator mat_eval_laplacian(mom_sources, mom,
                                                 laplacian_funtion);
      FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
          Laplacian_Interior;
      Laplacian_Interior.init(hst_sources, hst, eta, threshold_laplacianKernel);
      T.tic();
      Laplacian_Interior.compress(mat_eval_laplacian);
      const auto& trips_Laplacian = Laplacian_Interior.triplets();
      std::cout << "anz laplacianKernel:            "
                << trips_Laplacian.size() / NPTS_INTERIOR << std::endl;
      Eigen::SparseMatrix<double> Lcomp_Interior(NPTS_INTERIOR, N);
      Lcomp_Interior.setFromTriplets(trips_Laplacian.begin(),
                                     trips_Laplacian.end());
      Lcomp_Interior.makeCompressed();
      Laplacian += Lcomp_Interior;

       ////////////// full eval
      auto Perm_int = FMCA::permutationMatrix(hst_sources);
      FMCA::Matrix Laplacian_Interior_eval = laplacian_funtion.eval(P_sources,P);
      Laplacian_eval += Laplacian_Interior_eval;

      FMCA::Matrix Laplacian_Interior_eval_permuted = Perm_int.transpose() * Laplacian_Interior_eval * Perm_N;
      Laplacian_eval_permuted += Laplacian_Interior_eval_permuted;
      //////////////////////////////////////////////////////////////////////////////
    }

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

    std::cout << "anz A:                          "
              << tripletList.size() / N << std::endl;


    // check eval
    FMCA::Matrix TLT_eval = hst_sources.sampletTransform(Laplacian_eval_permuted);
    TLT_eval = hst.sampletTransform(TLT_eval.transpose()).transpose();

    FMCA::Matrix TBT_eval = hst_sources_border.sampletTransform(K_bnd_n_eval_permuted);
    TBT_eval = hst.sampletTransform(TBT_eval.transpose()).transpose();

    std::cout << "Laplacian error eval                    " << (Laplacian.toDense() - TLT_eval).norm() / TLT_eval.norm() << std::endl;
    std::cout << "Border error eval                       " << (Kcomp_border.toDense() - TBT_eval).norm() / TBT_eval.norm()
              << std::endl;


    FMCA::Matrix A_eval_samplets(N, N);
    A_eval_samplets << TLT_eval, TBT_eval;
    std::cout << "Matrix system error                     " << (A_eval_samplets - A.toDense()).norm()/A_eval_samplets.norm() << std::endl;
    //////////////////////////////////////////////////////////////////////////////
    // solve the dense system
    FMCA::Matrix A_eval(N, N);
    A_eval << Laplacian_eval, K_bnd_n_eval;

    FMCA::Vector b_eval(N);
    b_eval << f, u_bc;

    // Solver
    Eigen::ColPivHouseholderQR<FMCA::Matrix> solver_eval;
    solver_eval.compute(A_eval);
    FMCA::Vector alpha_eval =
        solver_eval.solve(b_eval);
    std::cout << "residual error full:                    "
              << (A_eval * alpha_eval - b_eval).norm() << std::endl;
    //////////////////////////////////////////////////////////////////////////////

    FMCA::Vector f_reordered = hst_sources.toClusterOrder(f);  
    FMCA::Vector f_transformed = hst_sources.sampletTransform(f_reordered);

    FMCA::Vector u_bc_reordered = hst_sources_border.toClusterOrder(u_bc);  
    FMCA::Vector u_bc_transformed = hst_sources_border.sampletTransform(u_bc_reordered);

    // Concatenate f vector and u_bc vector vertically
    Eigen::VectorXd b(N);
    b << f_transformed, u_bc_transformed;

    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::MetisOrdering<int>>
        Solver;
    Solver.analyzePattern(A);
    Solver.factorize(A);
    if (Solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomposition failed");
    }

    FMCA::Vector alpha = Solver.solve(b);
    std::cout << "residual error:                         "
              << (A * alpha - b).norm() << std::endl;
    
    FMCA::Vector alpha_inverseTransform = hst.inverseSampletTransform(alpha);
    FMCA::Vector alpha_permuted = hst.toNaturalOrder(alpha_inverseTransform);

    FMCA::Matrix K_eval_n = kernel_funtion_border.eval(Peval, P);

    std::cout << "K * alpha_dense - K * alpha_compressed     " << (K_eval_n * alpha_eval - K_eval_n* alpha_permuted).norm() / 
                                        (K_eval_n*alpha_eval).norm() <<std::endl;


    FMCA::Vector analytical_sol(Peval.cols());
    for (int i = 0; i < Peval.cols(); ++i) {
      double x = Peval(0, i);
      double y = Peval(1, i);
      double z = Peval(2, i);
      analytical_sol[i] = 1 - x * x - y * y - z * z;
    }

    std::cout << "error compressed    " << (analytical_sol - K_eval_n* alpha_permuted).norm() / 
                                        (K_eval_n*alpha_permuted).norm() <<std::endl;

    std::cout << "error full          " << (analytical_sol - K_eval_n* alpha_eval).norm() / 
                                        (K_eval_n*alpha_eval).norm() <<std::endl;


  std::cout << std::string(80, '-') << std::endl;
  return 0;
}
