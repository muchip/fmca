#include <algorithm>
#include <cmath>  // Include cmath for sqrt function
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <vector>  // Include vector for std::vector
//////////////////////////////////////////////////////////
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
#include "../FMCA/src/util/Plotter3D.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "../TestPDE/read_files_txt.h"
#include "MultiGridFunctions.h"

#define DIM 3

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

// Scalar averageCoordinates(const Eigen::VectorXd& Points) {
//   return Points.mean();
// }
// Scalar CustomC2Function(Scalar x, Scalar y, Scalar z, Scalar x0, Scalar y0,
//                         Scalar z0, Scalar x1, Scalar y1, Scalar z1, Scalar A,
//                         Scalar B, Scalar C, Scalar sigma, Scalar wx, Scalar
//                         wy, Scalar wz, Scalar r) {
//
//   Scalar gaussian = A * std::exp(-((x - x0) * (x - x0) + (y - y0) * (y - y0)
//   +
//                                    (z - z0) * (z - z0)) /
//                                  (sigma * sigma));
//   Scalar trigonometric =
//       B * std::sin(wx * x) * std::cos(wy * y) * std::sin(wz * z);
//   Scalar rational =
//       C * (1.0 / (1.0 + ((x - x1) * (x - x1) + (y - y1) * (y - y1) +
//                          (z - z1) * (z - z1)) /
//                             (r * r)));
//    return gaussian + trigonometric + rational;
// }

// Vector evalFunction(const Matrix& Points, Scalar x0, Scalar y0, Scalar z0,
//                     Scalar x1, Scalar y1, Scalar z1, Scalar A, Scalar B,
//                     Scalar C, Scalar sigma, Scalar wx, Scalar wy, Scalar wz,
//                     Scalar r) {
//   Vector f_values(Points.cols());
//   for (Eigen::Index i = 0; i < Points.cols(); ++i) {
//     f_values(i) =
//         CustomC2Function(Points(0, i), Points(1, i), Points(2, i), x0, y0,
//         z0,
//                          x1, y1, z1, A, B, C, sigma, wx, wy, wz, r);
//   }
//   return f_values;
// }

Scalar InkDiffusionFunction(Scalar x, Scalar y, Scalar z,
                            const std::vector<Eigen::Vector3d>& inkSources,
                            const std::vector<Scalar>& intensities,
                            const std::vector<Scalar>& spreads, Scalar time) {
  Scalar result = 0.0;

  for (size_t i = 0; i < inkSources.size(); ++i) {
    Scalar dx = x - inkSources[i](0);
    Scalar dy = y - inkSources[i](1);
    Scalar dz = z - inkSources[i](2);
    Scalar distSquared = dx * dx + dy * dy + dz * dz;
    Scalar currentSpread = spreads[i] * std::sqrt((1 + time));
    // Gaussian diffusion
    result += intensities[i] *
              std::exp(-distSquared / (2 * currentSpread * currentSpread));
  }

  return result;
}

Vector evalFunction(const Matrix& Points,
                        const std::vector<Eigen::Vector3d>& inkSources,
                        const std::vector<Scalar>& intensities,
                        const std::vector<Scalar>& spreads, Scalar time) {
  Vector f_values(Points.cols());
  for (Eigen::Index i = 0; i < Points.cols(); ++i) {
    f_values(i) = InkDiffusionFunction(Points(0, i), Points(1, i), Points(2, i),
                                       inkSources, intensities, spreads, time);
  }
  return f_values;
}

int main() {
  Tictoc T;
  ///////////////////////////////// Inputs: points
  Matrix P1, P2, P3, P4, P5, P6;
  readTXT("data/octopus_level1.txt", P1, DIM);
  readTXT("data/octopus_level2.txt", P2, DIM);
  readTXT("data/octopus_level3.txt", P3, DIM);
  readTXT("data/octopus_level4.txt", P4, DIM);
  readTXT("data/octopus_level5.txt", P5, DIM);
  readTXT("data/octopus_level6.txt", P6, DIM);

  // Output the cardinality of each normalized matrix
  std::cout << "Cardinality P1      " << P1.cols() << std::endl;
  std::cout << "Cardinality P2      " << P2.cols() << std::endl;
  std::cout << "Cardinality P3      " << P3.cols() << std::endl;
  std::cout << "Cardinality P4      " << P4.cols() << std::endl;
  std::cout << "Cardinality P5      " << P5.cols() << std::endl;
  std::cout << "Cardinality P6      " << P6.cols() << std::endl;

  Matrix Peval;
  readTXT("data/octopus_eval.txt", Peval, DIM);
  std::cout << "Cardinality Peval   " << Peval.cols() << std::endl;

  ///////////////////////////////// Nested cardinality of points
  std::vector<Matrix> P_Matrices = {P1, P2, P3, P4, P5, P6};
  int max_level = P_Matrices.size();

  ///////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Eigen::Index dtilde = 3;
  const Scalar threshold_kernel = 1e-6;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = 2 * (dtilde - 1);
  const std::string kernel_type = "matern32";
  const Scalar nu = 0.6;

  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "threshold_kernel    " << threshold_kernel << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;
  std::cout << "kernel_type         " << kernel_type << std::endl;
  std::cout << "nu                  " << nu << std::endl;

  ///////////////////////////////// Rhs
  std::vector<Vector> residuals;
  //   Scalar x1 = 0.;
  //   Scalar y1 = -0.1;
  //   Scalar z1 = -0.1;
  //   Scalar x0 = averageCoordinates(P6.row(0));
  //   Scalar y0 = averageCoordinates(P6.row(1));
  //   Scalar z0 = averageCoordinates(P6.row(2));
  //   Scalar A = 0.5;
  //   Scalar B = 0.3;
  //   Scalar C = 2.0;
  //   Scalar sigma = 0.1;
  //   Scalar wx = 10;
  //   Scalar wy = 10;
  //   Scalar wz = 10;
  //   Scalar r = 0.05;
  std::vector<Eigen::Vector3d> inkSources = {
      Eigen::Vector3d(-0.126, 0.4048, -0.0972),    
      Eigen::Vector3d(0.531, -0.0810, 0.0855) 
  };
  std::vector<Scalar> intensities = {10.0, 5.0};  // Intensity of each source
  std::vector<Scalar> spreads = {0.3, 0.3};    // Initial spread of each source
  Scalar time = 0.8;                             // Time since ink release

  ///////////////////////////////// Fill Distances and Residuals
  Vector fill_distances(max_level);
  for (Eigen::Index i = 0; i < max_level; ++i) {
    const Moments mom(P_Matrices[i], mpole_deg);
    const SampletMoments samp_mom(P_Matrices[i], dtilde - 1);
    const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[i]);
    Vector minDistance = minDistanceVector(hst, P_Matrices[i]);
    Scalar h = minDistance.maxCoeff();
    fill_distances[i] = h;

    Vector residual = evalFunction(P_Matrices[i], inkSources, intensities, spreads, time);
    residual = hst.toClusterOrder(residual);
    residual = hst.sampletTransform(residual);
    residuals.push_back(residual);
  }
  std::cout << fill_distances << std::endl;

  ///////////////////////////////// Coeffs Initialization
  std::vector<Vector> ALPHA;
  std::string base_filename_residuals = "Plots/ResidualsOctupus";

  ///////////////////////////////// Resolution --> Scheme = Matricial form
  for (Eigen::Index l = 0; l < max_level; ++l) {
    std::cout << "-------- Level " << l + 1 << " --------" << std::endl;
    std::cout << "Fill distance                      " << fill_distances[l]
              << std::endl;
    int n_pts = P_Matrices[l].cols();
    std::cout << "Number of points                   " << n_pts << std::endl;

    const Moments mom(P_Matrices[l], mpole_deg);
    const SampletMoments samp_mom(P_Matrices[l], dtilde - 1);
    const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[l]);

    for (Eigen::Index j = 0; j < l; ++j) {
      int n_pts_B = P_Matrices[j].cols();
      FMCA::Scalar sigma_B = nu * fill_distances[j];
      const CovarianceKernel kernel_funtion_B(kernel_type, sigma_B);
      T.tic();
      // Scope block for B_comp
      {
        Eigen::SparseMatrix<double> B_comp =
            UnsymmetricCompressor(mom, samp_mom, hst, kernel_funtion_B, eta,
                                  threshold_kernel / sigma_B, mpole_deg, dtilde,
                                  P_Matrices[l], P_Matrices[j]);
        Scalar compression_time_unsymmetric = T.toc();
        std::cout << "Compression time                   "
                  << compression_time_unsymmetric << std::endl;
        residuals[l] -= B_comp * ALPHA[j];
        std::cout << "Residuals updated" << std::endl;
      }  // B_comp goes out of scope and is destroyed here
    }

    ///////////////////////////////// Plot the residuals
    {
      Vector residual_natural_basis = hst.inverseSampletTransform(residuals[l]);
      residual_natural_basis = hst.toNaturalOrder(residual_natural_basis);
      // Create the filename for the residual
      std::ostringstream oss;
      oss << base_filename_residuals << "_level_" << l << ".vtk";
      std::string filename_residuals = oss.str();
      Plotter3D plotter;
      plotter.plotFunction(filename_residuals, P_Matrices[l],
                           residual_natural_basis);
    }
    Scalar sigma = nu * fill_distances[l];
    const CovarianceKernel kernel_funtion(kernel_type, sigma);
    T.tic();
    // Scope block for A_comp
    {
      Eigen::SparseMatrix<double> A_comp = SymmetricCompressor(
          mom, samp_mom, hst, kernel_funtion, eta, threshold_kernel / sigma,
          mpole_deg, dtilde, P_Matrices[l]);
      Scalar compression_time = T.toc();
      std::cout << "Compression time                   " << compression_time
                << std::endl;

      ///////////////////////////////// Solve the linear system
      Vector rhs = residuals[l];
      Vector alpha =
          solveSystem(A_comp, rhs, "ConjugateGradientwithPreconditioner", 1e-8);
      ALPHA.push_back(alpha);
    }  // A_comp goes out of scope and is destroyed here
  }

  ///////////////////////////////// Final Evaluation
  const Moments mom(Peval, mpole_deg);
  const SampletMoments samp_mom(Peval, dtilde - 1);
  const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, Peval);
  Vector exact_sol = evalFunction(Peval, inkSources, intensities, spreads, time);
  std::cout << "norm exact solution: " << exact_sol.norm() << std::endl;
  Vector solution = Evaluate(
      mom, samp_mom, hst, kernel_type, P_Matrices, Peval, ALPHA, fill_distances,
      max_level, nu, eta, threshold_kernel, mpole_deg, dtilde, exact_sol, hst,
      "Plots/SolutionOctupus");  // Plots/SolutionBunny

  return 0;
}
