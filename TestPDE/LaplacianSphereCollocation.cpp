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
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d(0, 1);
  // Normal distribution with mean 0 and stddev 1
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
    file.close();
  } else {
    std::cerr << "Unable to open file " << filename << std::endl;
  }
}

// This funtion in the main generates the 100k interior MonteCarloPoints and 50k
// MonteCarlo boundary points. Then for the comvergence and the solution of the
// problem, we select a subset of these Points, to garantee the nested property.
/*
  int N_interior = 100000;
  FMCA::Matrix P_interior = MonteCarloPointsInterior(N_interior);
  saveMatrixToFile(P_interior.transpose(),"MC_Interior_Circle_collocation.txt");
  int N_bnd = 50000;
  FMCA::Matrix P_bnd = MonteCarloPointsBoundary(N_bnd);
  saveMatrixToFile(P_bnd.transpose(), "MC_Bnd_Circle_collocation.txt");
*/

int main() {
  FMCA::Tictoc T;
  FMCA::Matrix P_interior_full;
  FMCA::Matrix P_bnd_full;
  readTXT("data/MC_Interior_Circle_collocation.txt", P_interior_full, DIM);
  readTXT("data/MC_Bnd_Circle_collocation.txt", P_bnd_full, DIM);
  //////////////////////////////////////////////////////////////////////////////
  // Parameters
  const int NPTS_INTERIOR = 5000;
  const int NPTS_BORDER = 2000;
  const int NPTS_EVAL = 1000;
  const FMCA::Scalar eta = 1. / 3.;
  const FMCA::Index dtilde = 6;
  const FMCA::Scalar threshold_kernel = 1e-10;
  const FMCA::Scalar MPOLE_DEG = 2 * (dtilde - 1);
  const std::string kernel_type = "MATERN52";
  const std::string kernel_type_laplacian = "MATERN52_SECOND_DERIVATIVE";
  const std::string solName = "SphereCollocationNumeric.vtk";
  const std::string solNameAnalytical = "SphereCollocationAnalytic.vtk";

  //////////////////////////////////////////////////////////////////////////////
  // Points
  FMCA::Matrix P_interior = P_interior_full.leftCols(NPTS_INTERIOR);
  FMCA::IO::plotPoints("P_sphere_interior.vtk", P_interior);
  FMCA::Matrix P_boundary = P_bnd_full.leftCols(NPTS_BORDER);
  FMCA::IO::plotPoints("P_sphere_bnd.vtk", P_boundary);
  FMCA::Matrix P(DIM, NPTS_INTERIOR + NPTS_BORDER);
  P << P_interior, P_boundary;
  FMCA::IO::plotPoints("P_sphere.vtk", P);
  int N = P.cols();
  FMCA::Matrix Peval = P_interior_full.rightCols(NPTS_EVAL);
  //////////////////////////////////////////////////////////////////////////////
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
  // Initialization of quantities for the Samplets: Moments and Tree
  std::cout << std::string(80, '-') << std::endl;
  const Moments mom_interior(P_interior, MPOLE_DEG);
  const SampletMoments samp_mom_interior(P_interior, dtilde - 1);
  H2SampletTree hst_interior(mom_interior, samp_mom_interior, 0, P_interior);

  const Moments mom_boundary(P_boundary, MPOLE_DEG);
  const SampletMoments samp_mom_boundary(P_boundary, dtilde - 1);
  H2SampletTree hst_boundary(mom_boundary, samp_mom_boundary, 0, P_boundary);

  const Moments mom(P, MPOLE_DEG);
  const SampletMoments samp_mom(P, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, P);
  //////////////////////////////////////////////////////////////////////////////
  // Compute the fill distance and choose the lenghtscale parameter
  FMCA::Vector minDistance = minDistanceVector(hst, P);
  auto maxElementIterator =
      std::max_element(minDistance.begin(), minDistance.end());
  FMCA::Scalar sigma_h = *maxElementIterator;
  std::cout << "fill distance:              " << sigma_h << std::endl;

  FMCA::Scalar sigma = 0.1;
  // FMCA::Scalar sigma = 20 * sigma_h;
  const FMCA::Scalar threshold_laplacianKernel =
      threshold_kernel * sigma_h * sigma_h;
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Number of interior points   " << NPTS_INTERIOR << std::endl;
  std::cout << "Number of border points     " << NPTS_BORDER << std::endl;
  std::cout << "Total number of points      " << N << std::endl;
  std::cout << "sigma                       " << sigma << std::endl;
  std::cout << "threshold kernel            " << threshold_kernel << std::endl;
  std::cout << "threshold laplacian         " << threshold_laplacianKernel
            << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // Compute the Block of the matrix related to K(P_border,P)
  const FMCA::CovarianceKernel kernel_function(kernel_type, sigma);
  const usMatrixEvaluatorKernel mat_eval_boundary(mom_boundary, mom,
                                                  kernel_function);
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> K_border;
  K_border.init(hst_boundary, hst, eta, threshold_kernel);
  T.tic();
  K_border.compress(mat_eval_boundary);
  const auto& trips_border = K_border.triplets();

  ////////////////////////////////// compression error
  FMCA::Vector x(N), y1(NPTS_BORDER), y2(NPTS_BORDER);
  FMCA::Scalar err = 0;
  FMCA::Scalar nrm = 0;
  for (auto i = 0; i < 100; ++i) {
    FMCA::Index index = rand() % N;
    x.setZero();
    x(index) = 1;
    FMCA::Vector col =
        kernel_function.eval(P_boundary, P.col(hst.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst_boundary.indices(),
                                             hst_boundary.block_size()));
    x = hst.sampletTransform(x);
    y2.setZero();
    for (const auto& i : trips_border) {
      y2(i.row()) += i.value() * x(i.col());
    }
    y2 = hst_boundary.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  std::cout << "compression error kernel    " << sqrt(err / nrm) << std::endl;

  std::cout << "anz kernel                  "
            << trips_border.size() / NPTS_BORDER << std::endl;
  Eigen::SparseMatrix<double> Kcomp_border(NPTS_BORDER, N);
  Kcomp_border.setFromTriplets(trips_border.begin(), trips_border.end());
  Kcomp_border.makeCompressed();
  //////////////////////////////////////////////////////////////////////////////
  // Compute the Block of the matrix related to Laplacian[K](P_interior,P)
  Eigen::SparseMatrix<double> Laplacian(NPTS_INTERIOR, N);
  Laplacian.setZero();
  for (int i = 0; i < DIM; ++i) {
    const FMCA::GradKernel laplacian_funtion(kernel_type_laplacian, sigma, 1,
                                             i);
    const usMatrixEvaluator mat_eval_laplacian(mom_interior, mom,
                                               laplacian_funtion);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
        Laplacian_Interior;
    Laplacian_Interior.init(hst_interior, hst, eta, threshold_laplacianKernel);
    Laplacian_Interior.compress(mat_eval_laplacian);
    const auto& trips_Laplacian = Laplacian_Interior.triplets();

    ////////////////////////////////// compression error
    FMCA::Vector x(N), y1(NPTS_INTERIOR), y2(NPTS_INTERIOR);
    FMCA::Scalar err = 0;
    FMCA::Scalar nrm = 0;
    for (auto i = 0; i < 100; ++i) {
      FMCA::Index index = rand() % N;
      x.setZero();
      x(index) = 1;
      FMCA::Vector col =
          laplacian_funtion.eval(P_interior, P.col(hst.indices()[index]));
      y1 = col(Eigen::Map<const FMCA::iVector>(hst_interior.indices(),
                                               hst_interior.block_size()));
      x = hst.sampletTransform(x);
      y2.setZero();
      for (const auto& i : trips_Laplacian) {
        y2(i.row()) += i.value() * x(i.col());
      }
      y2 = hst_interior.inverseSampletTransform(y2);
      err += (y1 - y2).squaredNorm();
      nrm += y1.squaredNorm();
    }
    std::cout << "compr error laplacian       " << sqrt(err / nrm) << std::endl;
    ;

    std::cout << "anz laplacianKernel         "
              << trips_Laplacian.size() / NPTS_INTERIOR << std::endl;
    Eigen::SparseMatrix<double> Lcomp_Interior(NPTS_INTERIOR, N);
    Lcomp_Interior.setFromTriplets(trips_Laplacian.begin(),
                                   trips_Laplacian.end());
    Lcomp_Interior.makeCompressed();
    Laplacian += Lcomp_Interior;
  }
  //////////////////////////////////////////////////////////////////////////////
  // Concatenate Laplacian[K](P_interior,P) and K(P_border,P) vertically
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
      tripletList.push_back(Eigen::Triplet<double>(it.row() + Laplacian.rows(),
                                                   it.col(), it.value()));
    }
  }

  Eigen::SparseMatrix<double> A(N, N);
  A.setFromTriplets(tripletList.begin(), tripletList.end());
  A.makeCompressed();
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "anz Matrix system            " << tripletList.size() / N
            << std::endl;

  std::vector<Eigen::Triplet<double>> filteredTriplets;
  for (int k = 0; k < A.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
      if (it.row() == it.col()) {
        filteredTriplets.emplace_back(it.row(), it.col(), it.value());
      } else {
        if (std::abs(it.value()) >= 1e-4) {
          filteredTriplets.emplace_back(it.row(), it.col(), it.value());
        }
      }
    }
  }

  Eigen::SparseMatrix<double> A_filtered(N, N);
  A_filtered.setFromTriplets(filteredTriplets.begin(), filteredTriplets.end());
  A_filtered.makeCompressed();
  std::cout << "anz Matrix system filtered     " << filteredTriplets.size() / N
            << std::endl;
  std::cout << "Matrix - Matrix filtered       " << (A_filtered.toDense()-A.toDense()).norm()/A.toDense().norm()
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Reorder and Sampet transform the right hand side of the linear system
  FMCA::Vector f_reordered = hst_interior.toClusterOrder(f);
  FMCA::Vector f_transformed = hst_interior.sampletTransform(f_reordered);
  FMCA::Vector u_bc_reordered = hst_boundary.toClusterOrder(u_bc);
  FMCA::Vector u_bc_transformed = hst_boundary.sampletTransform(u_bc_reordered);

  // Concatenate f vector and u_bc vector vertically
  Eigen::VectorXd b(N);
  b << f_transformed, u_bc_transformed;
  //////////////////////////////////////////////////////////////////////////////
  // Solve the linear system
  Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::MetisOrdering<int>>
      Solver;
  Solver.analyzePattern(A);
  Solver.factorize(A);
  if (Solver.info() != Eigen::Success) {
    throw std::runtime_error("Decomposition failed");
  }
  FMCA::Vector alpha = Solver.solve(b);
  std::cout << "residual error solver        " << (A * alpha - b).norm()
            << std::endl;
  std::cout << std::string(80, '-') << std::endl;

  FMCA::Vector alpha_grid = hst.inverseSampletTransform(alpha);
  alpha_grid = hst.toNaturalOrder(alpha_grid);
  //////////////////////////////////////////////////////////////////////////////
  // Solve the linear system filtered
  Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::MetisOrdering<int>>
      SolverFiltered;
  SolverFiltered.analyzePattern(A_filtered);
  SolverFiltered.factorize(A_filtered);
  if (SolverFiltered.info() != Eigen::Success) {
    throw std::runtime_error("Decomposition failed");
  }
  FMCA::Vector alpha_filtered = SolverFiltered.solve(b);
  std::cout << "residual error filtered        " << (A_filtered * alpha_filtered - b).norm()
            << std::endl;
  std::cout << std::string(80, '-') << std::endl;

  FMCA::Vector alpha_grid_filtered = hst.inverseSampletTransform(alpha_filtered);
  alpha_grid_filtered = hst.toNaturalOrder(alpha_grid_filtered);
  //////////////////////////////////////////////////////////////////////////////
  // Plot and check the solution
  {
    const Moments cmom(P, MPOLE_DEG);
    const Moments rmom(Peval, MPOLE_DEG);
    const H2ClusterTree hct(cmom, 0, P);
    const H2ClusterTree hct_eval(rmom, 0, Peval);
    const FMCA::Vector sx = hct.toClusterOrder(alpha_grid);
    const usMatrixEvaluatorKernel mat_eval(rmom, cmom, kernel_function);
    FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster> hmat;
    hmat.computePattern(hct_eval, hct, eta);
    // hmat.statistics();
    FMCA::Vector srec = hmat.action(mat_eval, sx);
    FMCA::Vector rec = hct_eval.toNaturalOrder(srec);
    FMCA::Plotter3D plotter;
    plotter.plotFunction(solName, Peval, rec);

    // Analytical sol of the problem
    FMCA::Vector analytical_sol(Peval.cols());
    for (int i = 0; i < Peval.cols(); ++i) {
      double x = Peval(0, i);
      double y = Peval(1, i);
      double z = Peval(2, i);
      analytical_sol[i] = 1 - x * x - y * y - z * z;
    }

    FMCA::Plotter3D plotter_analytical;
    plotter.plotFunction(solNameAnalytical, Peval, analytical_sol);

    FMCA::Scalar error = (rec - analytical_sol).norm() / analytical_sol.norm();
    std::cout << "error                       " << error << std::endl;
  }

    // Plot and check the solution
  {
    const Moments cmom(P, MPOLE_DEG);
    const Moments rmom(Peval, MPOLE_DEG);
    const H2ClusterTree hct(cmom, 0, P);
    const H2ClusterTree hct_eval(rmom, 0, Peval);
    const FMCA::Vector sx = hct.toClusterOrder(alpha_grid_filtered);
    const usMatrixEvaluatorKernel mat_eval(rmom, cmom, kernel_function);
    FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster> hmat;
    hmat.computePattern(hct_eval, hct, eta);
    // hmat.statistics();
    FMCA::Vector srec = hmat.action(mat_eval, sx);
    FMCA::Vector rec = hct_eval.toNaturalOrder(srec);

    // Analytical sol of the problem
    FMCA::Vector analytical_sol(Peval.cols());
    for (int i = 0; i < Peval.cols(); ++i) {
      double x = Peval(0, i);
      double y = Peval(1, i);
      double z = Peval(2, i);
      analytical_sol[i] = 1 - x * x - y * y - z * z;
    }


    FMCA::Scalar error_filtered = (rec - analytical_sol).norm() / analytical_sol.norm();
    std::cout << "error_filtered               " << error_filtered << std::endl;
  }

  std::cout << std::string(80, '-') << std::endl;
  return 0;
}
