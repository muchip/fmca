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
  Matrix P9;
  readTXT("data/square01_uniform_grid_level9.txt", P9, DIM);
  std::cout << "Cardinality P9      " << P9.cols() << std::endl;

  Matrix P10;
  readTXT("data/square01_uniform_grid_level10.txt", P10, DIM);
  std::cout << "Cardinality P10     " << P10.cols() << std::endl;

  std::vector<Scalar> dtildes = {5, 6, 7, 8};
  for (Scalar dtilde : dtildes) {
    ///////////////////////////////// Parameters
    const Scalar eta = 1. / DIM;
    // const Scalar dtilde = 5;
    const Scalar threshold_kernel = 1e-8;
    const Scalar threshold_weights = 0;
    const Scalar mpole_deg = 2 * (dtilde - 1);
    const std::string kernel_type = "matern32";
    const Scalar nu = 3;
    std::cout << "--------------------" << std::endl;
    std::cout << "dtilde              " << dtilde << std::endl;
    std::cout << "threshold_kernel    " << threshold_kernel << std::endl;
    std::cout << "kernel type         " << kernel_type << std::endl;
    std::cout << "nu                  " << nu << std::endl;

    ///////////////////////////////// Rhs
    std::vector<Matrix> P_Matrices = {P9, P10};
    int max_level = P_Matrices.size();
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

      Vector residual = evalFrankeFunction(P_Matrices[i]);
      residual = hst.toClusterOrder(residual);
      residual = hst.sampletTransform(residual);
      residuals.push_back(residual);
    }

    ///////////////////////////////// Fill Distances and Residuals
    const Moments mom_P9(P9, mpole_deg);
    const SampletMoments samp_mom_P9(P9, dtilde - 1);
    const H2SampletTree<ClusterTree> hst_P9(mom_P9, samp_mom_P9, 0, P9);

    Vector minDistance = minDistanceVector(hst_P9, P9);
    Scalar fill_distance_P9 = minDistance.maxCoeff();

    const Moments mom_P10(P10, mpole_deg);
    const SampletMoments samp_mom_P10(P10, dtilde - 1);
    const H2SampletTree<ClusterTree> hst_P10(mom_P10, samp_mom_P10, 0, P10);

    Vector minDistance_P10 = minDistanceVector(hst_P10, P10);
    Scalar fill_distance_P10 = minDistance_P10.maxCoeff();

    std::cout << "Fill distance P9      " << fill_distance_P9 << std::endl;
    std::cout << "Fill distance P10     " << fill_distance_P10 << std::endl;

    {
      ///////////////////////////////// Unsymmetric Compression
      Scalar sigma_P9 = nu * fill_distance_P9;
      const CovarianceKernel kernel_function_P9(kernel_type, sigma_P9);
      T.tic();
      Eigen::SparseMatrix<double> B_comp = UnsymmetricCompressor(
          mom_P10, samp_mom_P10, hst_P10, kernel_function_P9, eta,
          threshold_kernel / sigma_P9, mpole_deg, dtilde, P10, P9);
      Scalar time_unsymmetric = T.toc();
      std::cout << "Dimensions = " << P10.cols() << " x " << P9.cols();
      std::cout << "Time unsymmetric compression min = "
                << time_unsymmetric / 60 << std::endl;
    }

    {
      ///////////////////////////////// Symmetric Compression
      Scalar sigma_P10 = nu * fill_distance_P10;
      const CovarianceKernel kernel_function_P10(kernel_type, sigma_P10);
      T.tic();
      Eigen::SparseMatrix<double> A_comp = SymmetricCompressor(
          mom_P10, samp_mom_P10, hst_P10, kernel_function_P10, eta,
          threshold_kernel / sigma_P10, mpole_deg, dtilde, P10);
      Scalar time_symmetric = T.toc();
      std::cout << "Dimensions = " << P10.cols() << " x " << P10.cols();
      std::cout << "Time symmetric compression min = " << time_symmetric / 60
                << std::endl;
    }
  }
  std::cout << "------------------------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------------------------" << std::endl;
  return 0;
}
