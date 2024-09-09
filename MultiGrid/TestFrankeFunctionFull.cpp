#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
// ##############################
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

int main() {
  Tictoc T;
  ///////////////////////////////// Inputs: points + maximum level
  Matrix P0;
  readTXT("data/square01_2k.txt", P0, DIM);
  std::cout << "Cardianlity P0 =     " << P0.cols() << std::endl;
  Matrix P1;
  readTXT("data/square01_8k.txt", P1, DIM);
  std::cout << "Cardianlity P1 =     " << P1.cols() << std::endl;
  Matrix P2;
  readTXT("data/square01_33k.txt", P2, DIM);
  std::cout << "Cardianlity P2 =     " << P2.cols() << std::endl;
  Matrix Peval;
  readTXT("data/uniform_vertices_UnitSquare_10k.txt", Peval, DIM);
  std::cout << "Cardianlity Peval =  " << Peval.cols() << std::endl;
  ///////////////////////////////// Nested cardinality of points
  std::vector<Matrix> P_Matrices = {P0, P1};
  int max_level = P_Matrices.size();
  ///////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 6;
  const Scalar threshold_kernel = 1e-8;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = 2 * (dtilde - 1);
  const std::string kernel_type = "matern32";
  const Scalar mu = 1;
  ///////////////////////////////// Fill Distances and Residuals
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
    residuals.push_back(residual);
  }
  ///////////////////////////////// Coeffs Initialization
  std::vector<Vector> ALPHA;

  ///////////////////////////////// Resolution --> Scheme = Matricial form
  for (Index l = 0; l < max_level; ++l) {
    std::cout << "-------- Level " << l << " --------" << std::endl;
    std::cout << "Fill distance        " << fill_distances[l] << std::endl;
    int n_pts = P_Matrices[l].cols();
    std::cout << "Number of points     " << n_pts << std::endl;

    for (Index j = 0; j < l; ++j) {
      FMCA::Scalar sigma_B = mu * fill_distances[j];
      const CovarianceKernel kernel_funtion_B(kernel_type, sigma_B);
      Matrix B = kernel_funtion_B.eval(P_Matrices[l], P_Matrices[j]);
      residuals[l] -= B * ALPHA[j];
    }

    Scalar sigma = mu * fill_distances[l];
    const CovarianceKernel kernel_funtion(kernel_type, sigma);
    Matrix A = kernel_funtion.eval(P_Matrices[l], P_Matrices[l]);

    ///////////////////////////////// Solve the linear system
    Vector rhs = residuals[l];
    Eigen::ColPivHouseholderQR<Matrix> solver;
    solver.compute(A);
    Vector alpha = solver.solve(rhs);
    std::cout << "Residual error:          " << (A * alpha - rhs).norm()
              << std::endl;

    ALPHA.push_back(alpha);
  }

  ///////////////////////////////// Final Evaluation
  Vector solution(Peval.cols());
  // Eigen::VectorXd::Zero(Peval.cols());
  Vector exact_sol = evalFrankeFunction(Peval);

  for (Index i = 0; i < max_level; ++i) {
    std::cout << "Evaluation level " << i << std::endl;
    Scalar sigma = mu * fill_distances[i];
    const CovarianceKernel kernel_funtion(kernel_type, sigma);
    Matrix K = kernel_funtion.eval(Peval, P_Matrices[i]);
    solution += K * ALPHA[i];

    std::cout << "Error: " << (solution - exact_sol).norm() / exact_sol.norm()
            << std::endl;
  }
  ///////////////////////////////// Error
//   Vector exact_sol = evalFrankeFunction(Peval);
//   std::cout << "Error: " << (solution - exact_sol).norm() / exact_sol.norm()
//             << std::endl;

  return 0;
}
