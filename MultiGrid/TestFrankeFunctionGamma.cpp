#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

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

Scalar FrankeFunction(Scalar x, Scalar y) {
  Scalar term1 = (3.0 / 4.0) * exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0) / 4.0) -
                                   ((9.0 * y - 2.0) * (9.0 * y - 2.0) / 4.0));
  Scalar term2 = (3.0 / 4.0) * exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0) / 49.0) -
                                   ((9.0 * y - 2.0) * (9.0 * y - 2.0) / 10.0));
  Scalar term3 = (1.0 / 2.0) * exp(-((9.0 * x - 7.0) * (9.0 * x - 7.0) / 4.0) -
                                   ((9.0 * y - 3.0) * (9.0 * y - 3.0) / 4.0));
  Scalar term4 = (1.0 / 5.0) * exp(-((9.0 * x - 4.0) * (9.0 * x - 4.0)) -
                                   ((9.0 * y - 7.0) * (9.0 * y - 7.0)));
  return term1 + term2 + term3 - term4;
}

Vector evalFrankeFunction(Matrix Points) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = FrankeFunction(Points(0, i), Points(1, i));
  }
  return f;
}

void runForMu(Scalar nu) {
  // Redirect std::cout to a file
  std::ostringstream filename;
  filename << "ResultsFranke_nu" << nu << "_Matern52_threshold_aPost_10-6_noIDX.txt";
  std::ofstream out(filename.str());
  std::streambuf *coutbuf = std::cout.rdbuf();  // Save old buf
  std::cout.rdbuf(out.rdbuf());                 // Redirect std::cout to out

  Tictoc T;
  ///////////////////////////////// Inputs: points + maximum level
  Matrix P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, Peval;
  readTXT("data/square01_uniform_grid_level1.txt", P1, DIM);
  readTXT("data/square01_uniform_grid_level2.txt", P2, DIM);
  readTXT("data/square01_uniform_grid_level3.txt", P3, DIM);
  readTXT("data/square01_uniform_grid_level4.txt", P4, DIM);
  readTXT("data/square01_uniform_grid_level5.txt", P5, DIM);
  readTXT("data/square01_uniform_grid_level6.txt", P6, DIM);
  readTXT("data/square01_uniform_grid_level7.txt", P7, DIM);
  readTXT("data/square01_uniform_grid_level8.txt", P8, DIM);
  readTXT("data/square01_uniform_grid_level9.txt", P9, DIM);
  readTXT("data/square01_uniform_grid_level10.txt", P10, DIM);
  readTXT("data/uniform_vertices_UnitSquare_40k.txt", Peval, DIM);

  ///////////////////////////////// Nested cardinality of points
  std::vector<Matrix> P_Matrices = {P1, P2, P3, P4, P5, P6, P7, P8, P9, P10};
  int max_level = P_Matrices.size();
  ///////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar threshold_kernel = 1e-10;
  const Scalar threshold_aPost = 1e-6;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = 2 * (dtilde - 1);
  const std::string kernel_type = "matern52";
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "threshold_kernel    " << threshold_kernel << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;
  std::cout << "kernel_type         " << kernel_type << std::endl;
  std::cout << "nu                  " << nu << std::endl;

  ///////////////////////////////// Rhs
  std::vector<Vector> residuals;
  Vector fill_distances(max_level);
  for (Index i = 0; i < max_level; ++i) {
    const Moments mom(P_Matrices[i], mpole_deg);
    const SampletMoments samp_mom(P_Matrices[i], dtilde - 1);
    const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[i]);
    Vector minDistance = minDistanceVector(hst, P_Matrices[i]);
    Scalar h = minDistance.maxCoeff();
    fill_distances[i] = h;

    Vector residual = evalFrankeFunction(P_Matrices[i]);
    residual = hst.toClusterOrder(residual);
    residual = hst.sampletTransform(residual);
    residuals.push_back(residual);
  }

  ///////////////////////////////// Coeffs Initialization
  std::vector<Vector> ALPHA;

  ///////////////////////////////// Resolution --> Scheme = Matricial form
  for (Index l = 0; l < max_level; ++l) {
    std::cout << "-------- Level " << l + 1 << " --------" << std::endl;
    std::cout << "Fill distance                      " << fill_distances[l]
              << std::endl;
    int n_pts = P_Matrices[l].cols();
    std::cout << "Number of points                   " << n_pts << std::endl;

    const Moments mom(P_Matrices[l], mpole_deg);
    const SampletMoments samp_mom(P_Matrices[l], dtilde - 1);
    const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[l]);

    for (Index j = 0; j < l; ++j) {
      int n_pts_B = P_Matrices[j].cols();
      FMCA::Scalar sigma_B = nu * fill_distances[j];
      const CovarianceKernel kernel_funtion_B(kernel_type, sigma_B);
      {
        Eigen::SparseMatrix<double> B_comp = UnsymmetricCompressor(
            mom, samp_mom, hst, kernel_funtion_B, eta,
            threshold_kernel, threshold_aPost, mpole_deg, dtilde,
            P_Matrices[l], P_Matrices[j]);
        residuals[l] -= B_comp * ALPHA[j];
      }  // B_comp goes out of scope and is destroyed here
      std::cout << "------------------------------------------" << std::endl;
    }

    Scalar sigma = nu * fill_distances[l];
    const CovarianceKernel kernel_funtion(kernel_type, sigma);
    {
      Eigen::SparseMatrix<double> A_comp = SymmetricCompressor(
          mom, samp_mom, hst, kernel_funtion, eta, threshold_kernel, threshold_aPost,
          mpole_deg, dtilde, P_Matrices[l]);
      std::cout << "------------------------------------------" << std::endl;

      Vector rhs = residuals[l];
      Vector alpha = solveSystem(A_comp, rhs, "ConjugateGradient", 1e-6);
      ALPHA.push_back(alpha);
      std::cout << "------------------------------------------" << std::endl;
    }  // A_comp goes out of scope and is destroyed here
  }

  const Moments mom(Peval, mpole_deg);
  const SampletMoments samp_mom(Peval, dtilde - 1);
  const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, Peval);
  Vector exact_sol = evalFrankeFunction(Peval);
  Vector solution = Evaluate(
      mom, samp_mom, hst, kernel_type, P_Matrices, Peval, ALPHA, fill_distances,
      max_level, nu, eta, threshold_kernel, threshold_aPost, mpole_deg, dtilde, exact_sol, hst,
      "");  //"Plots/SolutionFranke"
  std::cout << "------------------------------------------" << std::endl;
  std::cout << "------------------------------------------" << std::endl;
  // Restore std::cout to its original state
  std::cout.rdbuf(coutbuf);
}

int main() {
  std::vector<Scalar> nus = {0.5, 1, 1.5, 2};
  for (Scalar nu :  nus) {
    runForMu(nu);
  }
  return 0;
}
