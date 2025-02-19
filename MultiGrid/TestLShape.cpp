#include <math.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
// ##############################
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Plotter.h"
#include "../FMCA/src/util/Tictoc.h"
#include "read_files_txt.h"
#include "MultiGridFunctions.h"

#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluatorKernel =
    FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluatorKernel =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using EigenCholesky =
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>;

using namespace FMCA;

Scalar AnalyticalSolution(Scalar x, Scalar y) {
  Scalar r = sqrt(x * x + y * y);
  Scalar phi = atan2(y, x);
  if (phi <= 0) {
    phi += 2 * FMCA_PI;  // Adjust phi to be in [Pi/2,2*Pi]
  }
  if (r == 0) {
    return 0;
  }
  return -pow(r, 2.0 / 3.0) * sin((2 * phi - M_PI) / 3);
}

Vector evalAnalyticalSolution(Matrix Points) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = AnalyticalSolution(Points(0, i), Points(1, i));
  }
  return f;
}

int main() {
  Tictoc T;
  ///////////////////////////////// Inputs: points + maximum level
  Matrix P1;
  readTXT("data/L_shape_uniform_grid_level1.txt", P1, DIM);
  std::cout << "Cardianlity P1      " << P1.cols() << std::endl;
  Matrix P2;
  readTXT("data/L_shape_uniform_grid_level2.txt", P2, DIM);
  std::cout << "Cardianlity P2      " << P2.cols() << std::endl;
  Matrix P3;
  readTXT("data/L_shape_uniform_grid_level3.txt", P3, DIM);
  std::cout << "Cardianlity P3      " << P3.cols() << std::endl;
  Matrix P4;
  readTXT("data/L_shape_uniform_grid_level4.txt", P4, DIM);
  std::cout << "Cardianlity P4      " << P4.cols() << std::endl;
  Matrix P5;
  readTXT("data/L_shape_uniform_grid_level5.txt", P5, DIM);
  std::cout << "Cardianlity P5      " << P5.cols() << std::endl;
  Matrix P6;
  readTXT("data/L_shape_uniform_grid_level6.txt", P6, DIM);
  std::cout << "Cardianlity P6      " << P6.cols() << std::endl;
  Matrix P7;
  readTXT("data/L_shape_uniform_grid_level7.txt", P7, DIM);
  std::cout << "Cardianlity P7      " << P7.cols() << std::endl;
  Matrix P8;
  readTXT("data/L_shape_uniform_grid_level8.txt", P8, DIM);
  std::cout << "Cardianlity P8      " << P8.cols() << std::endl;
  Matrix Peval;
  readTXT("data/L_shape_uniform_grid_level9.txt", Peval, DIM);
  std::cout << "Cardianlity Peval   " << Peval.cols() << std::endl;
  ///////////////////////////////// Nested cardinality of points
  std::vector<Matrix> P_Matrices = {P1, P2, P3, P4, P5, P6, P7, P8};
  int max_level = P_Matrices.size();
  ///////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar threshold_kernel = 1e-6;
  const Scalar threshold_aPost = -1;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = 2 * (dtilde - 1);
  const std::string kernel_type = "exponential";
  const Scalar nu = 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "threshold_kernel    " << threshold_kernel << std::endl;
  std::cout << "threshold_aPost     " << threshold_aPost << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;
  std::cout << "kernel_type         " << kernel_type << std::endl;
  std::cout << "nu                  " << nu << std::endl;
  ///////////////////////////////// Rhs
  std::vector<Vector> residuals;
  ///////////////////////////////// Fill Distances and Residuals
  Vector fill_distances(max_level);
  for (Index i = 0; i < max_level; ++i) {
    const Moments mom(P_Matrices[i], mpole_deg);
    const SampletMoments samp_mom(P_Matrices[i], dtilde - 1);
    const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[i]);
    Vector minDistance = minDistanceVector(hst, P_Matrices[i]);
    Scalar h = minDistance.maxCoeff();
    fill_distances[i] = h;

    Vector residual = evalAnalyticalSolution(P_Matrices[i]);
    residual = hst.toClusterOrder(residual);
    residual = hst.sampletTransform(residual);
    residuals.push_back(residual);
  }
  std::cout << fill_distances << std::endl;

  ///////////////////////////////// Coeffs Initialization
  std::vector<Vector> ALPHA;
  std::string base_filename_residuals = "Plots/LogResidualsLShape";
  ///////////////////////////////// Resolution --> Scheme = Matricial form
  for (Index l = 0; l < max_level; ++l) {
    std::cout << "-------- Level " << l + 1 << " --------" << std::endl;
    std::cout << "Fill distance                      " << fill_distances[l]
              << std::endl;
    int n_pts = P_Matrices[l].cols();
    std::cout << "Number of points                   " << n_pts << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    const Moments mom(P_Matrices[l], mpole_deg);
    const SampletMoments samp_mom(P_Matrices[l], dtilde - 1);
    const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[l]);

    for (Index j = 0; j < l; ++j) {
      int n_pts_B = P_Matrices[j].cols();
      FMCA::Scalar sigma_B = nu * fill_distances[j];
      const CovarianceKernel kernel_funtion_B(kernel_type, sigma_B);
      // Scope block for B_comp
      {
        Eigen::SparseMatrix<double> B_comp = UnsymmetricCompressor(
            mom, samp_mom, hst, kernel_funtion_B, eta, threshold_kernel, threshold_aPost,
            mpole_deg, dtilde, P_Matrices[l], P_Matrices[j]);
        T.tic();
        residuals[l] -= B_comp * ALPHA[j];
        T.toc("time residual update =           ");

        // std::ostringstream oss;
        // oss << "matlabPlots/B" << "_level_" << l << "_" << j;
        // std::string matrix_name = oss.str();
        // FMCA::IO::print2spascii(matrix_name, B_comp, "w");
      }  // B_comp goes out of scope and is destroyed here
      std::cout << "------------------------------------------" << std::endl;
    }
    ///////////////////////////////// Plot the residuals
    {
      Vector residual_natural_basis = hst.inverseSampletTransform(residuals[l]);
      residual_natural_basis = hst.toNaturalOrder(residual_natural_basis);
      Vector abs_residual(residual_natural_basis.size());
      for (Eigen::Index i = 0; i < residual_natural_basis.size(); ++i) {
          abs_residual[i] = std::log(std::abs(residual_natural_basis[i]));
      }
      // Create the filename for the residual
      std::ostringstream oss;
      oss << base_filename_residuals << "_level_" << l << ".vtk";
      std::string filename_residuals = oss.str();
      Plotter2D plotter;
      plotter.plotFunction2D(filename_residuals, P_Matrices[l],
                             abs_residual);
    }

    Scalar sigma = nu * fill_distances[l];
    const CovarianceKernel kernel_funtion(kernel_type, sigma);
    {
      Eigen::SparseMatrix<double> A_comp = SymmetricCompressor(
          mom, samp_mom, hst, kernel_funtion, eta, threshold_kernel, threshold_aPost, mpole_deg,
          dtilde, P_Matrices[l]);
      std::cout << "------------------------------------------" << std::endl;

      // std::ostringstream oss;
      // oss << "matlabPlots/A" << "_level_" << l << "_" << l;
      // std::string matrix_name = oss.str();
      // FMCA::IO::print2spascii(matrix_name, A_comp, "w");
      ///////////////////////////////// Solve the linear system
      Vector rhs = residuals[l];
      Vector alpha = solveSystem(A_comp, rhs, "ConjugateGradient", 1e-8);
      ALPHA.push_back(alpha);
      std::cout << "------------------------------------------" << std::endl;
    }  // A_comp goes out of scope and is destroyed here
  }

  ///////////////////////////////// Final Evaluation
  const Moments mom(Peval, mpole_deg);
  const SampletMoments samp_mom(Peval, dtilde - 1);
  const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, Peval);
  Vector exact_sol = evalAnalyticalSolution(Peval);
  Vector solution = Evaluate(
      mom, samp_mom, hst, kernel_type, P_Matrices, Peval, ALPHA, fill_distances,
      max_level, nu, eta, threshold_kernel, threshold_aPost, mpole_deg, dtilde, exact_sol, hst,
      "");  //"Plots/SolutionLShape"

  return 0;
}
