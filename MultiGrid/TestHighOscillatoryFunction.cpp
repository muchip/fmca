#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
// ##############################
#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
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
#include "../FMCA/src/util/permutation.h"
#include "../TestPDE/read_files_txt.h"
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

Scalar SinFunction(Scalar x, Scalar y) {
  return sin(1. / (x + 0.03)) * sin(1. / (0.5 * y + 0.03));
}

Vector evalSinFunction(Matrix Points) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = SinFunction(Points(0, i), Points(1, i));
  }
  return f;
}

int main() {
  Tictoc T;
  ///////////////////////////////// Inputs: points + maximum level
  Matrix P0;
  readTXT("data/square01_145.txt", P0, DIM);
  std::cout << "Cardianlity P0      " << P0.cols() << std::endl;
  Matrix P1;
  readTXT("data/square01_545.txt", P1, DIM);
  std::cout << "Cardianlity P1      " << P1.cols() << std::endl;
  Matrix P2;
  readTXT("data/square01_2k.txt", P2, DIM);
  std::cout << "Cardianlity P2      " << P2.cols() << std::endl;
  Matrix P3;
  readTXT("data/square01_8k.txt", P3, DIM);
  std::cout << "Cardianlity P3      " << P3.cols() << std::endl;
  Matrix P4;
  readTXT("data/square01_33k.txt", P4, DIM);
  std::cout << "Cardianlity P4      " << P4.cols() << std::endl;
  Matrix P5;
  readTXT("data/square01_130k.txt", P5, DIM);
  std::cout << "Cardianlity P5      " << P5.cols() << std::endl;
  Matrix P6;
  readTXT("data/square01_525k.txt", P6, DIM);
  std::cout << "Cardianlity P6      " << P6.cols() << std::endl;
  Matrix P7;
  readTXT("data/square01_2M.txt", P7, DIM);
  std::cout << "Cardianlity P7      " << P7.cols() << std::endl;
  Matrix Peval;
  readTXT("data/uniform_vertices_UnitSquare_40k.txt", Peval, DIM);
  std::cout << "Cardianlity Peval   " << Peval.cols() << std::endl;
  ///////////////////////////////// Nested cardinality of points
  std::vector<Matrix> P_Matrices = {P0, P1, P2, P3, P4, P5, P6, P7};
  int max_level = P_Matrices.size();
  ///////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 6;
  const Scalar threshold_kernel = 1e-8;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = 2 * (dtilde - 1);
  const std::string kernel_type = "matern32";
  const Scalar mu = 4;
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

    Vector residual = evalSinFunction(P_Matrices[i]);
    residual = hst.toClusterOrder(residual);
    residual = hst.sampletTransform(residual);
    residuals.push_back(residual);
  }
  ///////////////////////////////// Coeffs Initialization
  std::vector<Vector> ALPHA;
  std::string base_filename_residuals = "Plots/ResidualsSin";
  ///////////////////////////////// Resolution --> Scheme = Matricial form
  for (Index l = 0; l < max_level; ++l) {
    std::cout << "-------- Level " << l << " --------" << std::endl;
    std::cout << "Fill distance                      " << fill_distances[l]
              << std::endl;
    int n_pts = P_Matrices[l].cols();
    std::cout << "Number of points                   " << n_pts << std::endl;

    const Moments mom(P_Matrices[l], mpole_deg);
    const SampletMoments samp_mom(P_Matrices[l], dtilde - 1);
    const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[l]);

    for (Index j = 0; j < l; ++j) {
      int n_pts_B = P_Matrices[j].cols();
      Scalar sigma_B = mu * fill_distances[j];
      const CovarianceKernel kernel_funtion_B(kernel_type, sigma_B);
      Eigen::SparseMatrix<double> B_comp = UnsymmetricCompressor(
          kernel_funtion_B, eta, threshold_kernel, mpole_deg, dtilde,
          P_Matrices[l], P_Matrices[j]);
      residuals[l] -= B_comp * ALPHA[j];
    }
    ///////////////////////////////// Plot the residuals
    Vector residual_natural_basis = hst.inverseSampletTransform(residuals[l]);
    residual_natural_basis = hst.toNaturalOrder(residual_natural_basis);
    // Create the filename for the residual
    std::ostringstream oss;
    oss << base_filename_residuals << "_level_" << l << ".vtk";
    std::string filename_residuals = oss.str();
    Plotter2D plotter;
    plotter.plotFunction2D(filename_residuals, P_Matrices[l],
                           residual_natural_basis);

    Scalar sigma = mu * fill_distances[l];
    const CovarianceKernel kernel_funtion(kernel_type, sigma);
    Eigen::SparseMatrix<double> A_comp =
        SymmetricCompressor(kernel_funtion, eta, threshold_kernel, mpole_deg,
                            dtilde, P_Matrices[l]);

    ///////////////////////////////// Solve the linear system
    Vector rhs = residuals[l];
    Vector alpha = solveSystem(A_comp, rhs, "ConjugateGradient");
    //   Vector alpha = solveSystem(A_comp, rhs, "Cholmod");
    ALPHA.push_back(alpha);
  }

  ///////////////////////////////// Final Evaluation
  const Moments mom(Peval, mpole_deg);
  const SampletMoments samp_mom(Peval, dtilde - 1);
  const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, Peval);
  Vector exact_sol = evalSinFunction(Peval);
  Plotter2D plotter_exact;
  plotter_exact.plotFunction2D("Plots/SolutionSinExact", Peval, exact_sol);
  Vector solution = Evaluate(
      kernel_type, P_Matrices, Peval, ALPHA, fill_distances, max_level, mu, eta,
      threshold_kernel, mpole_deg, dtilde, exact_sol, hst, "Plots/SolutionSin");
  return 0;
}
