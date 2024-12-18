#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <math.h>
// ##############################
#include <Eigen/Sparse>
#include <Eigen/Dense>

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

int main() {
  Tictoc T;
  ///////////////////////////////// Inputs: points + maximum level
  Matrix P1;
  readTXT("data/L_shape_uniform_grid_level1.txt", P1, DIM);
  Matrix P2;
  readTXT("data/L_shape_uniform_grid_level2.txt", P2, DIM);
  Matrix P3;
  readTXT("data/L_shape_uniform_grid_level3.txt", P3, DIM);
  Matrix P4;
  readTXT("data/L_shape_uniform_grid_level4.txt", P4, DIM);
  Matrix P5;
  readTXT("data/L_shape_uniform_grid_level5.txt", P5, DIM);
  Matrix P6;
  readTXT("data/L_shape_uniform_grid_level6.txt", P6, DIM);
  Matrix P7;
  readTXT("data/L_shape_uniform_grid_level7.txt", P7, DIM);
  Matrix P8;
  readTXT("data/L_shape_uniform_grid_level8.txt", P8, DIM);
  Matrix Peval;
  readTXT("data/L_shape_uniform_grid_level9.txt", Peval, DIM);

  std::vector<Matrix> P_Matrices = {P1, P2, P3, P4, P5, P6, P7, P8};
  int max_level = P_Matrices.size();

  ///////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar threshold_kernel = 1e-8;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = 2 * (dtilde - 1);
  const std::string kernel_type = "exponential";
  const Scalar nu = 2;

  ///////////////////////////////// Fill Distances
  Vector fill_distances(max_level);
  for (Index i = 0; i < max_level; ++i) {
    const Moments mom(P_Matrices[i], mpole_deg);
    const SampletMoments samp_mom(P_Matrices[i], dtilde - 1);
    const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[i]);
    Vector minDistance = minDistanceVector(hst, P_Matrices[i]);
    Scalar h = minDistance.maxCoeff();
    fill_distances[i] = h;
  }
  std::cout << fill_distances << std::endl;
  ///////////////////////////////// Reference point
  Eigen::Vector2d reference_point;
  reference_point << -0.25, -0.25; // (-1/4, -1/4)

  ///////////////////////////////// Resolution --> Scheme = Matricial form
  for (Index l = 0; l < max_level; ++l) {
    std::cout << "-------- Level " << l + 1 << " --------" << std::endl;

    int n_pts = P_Matrices[l].cols();
    std::cout << "Number of points                   " << n_pts << std::endl;

    const Moments mom(P_Matrices[l], mpole_deg);
    const SampletMoments samp_mom(P_Matrices[l], dtilde - 1);
    const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[l]);

    Scalar sigma = nu * fill_distances[l];
    const CovarianceKernel kernel_function(kernel_type, sigma);

    Vector kernel_values(n_pts);
    // Evaluate kernel at each point with respect to the reference point
    for (Index i = 0; i < n_pts; ++i) {
      Eigen::Vector2d point = P_Matrices[l].col(i);
      Scalar distance = (point - reference_point).norm();
      kernel_values[i] = exp(-distance / sigma);
    }

    // Save the kernel values to a file
    std::ofstream outfile("matlabPlots/kernel_values_level" + std::to_string(l + 1) + ".txt");
    if (outfile.is_open()) {
      for (Index i = 0; i < n_pts; ++i) {
        outfile << kernel_values[i] << std::endl;
      }
      outfile.close();
    } else {
      std::cerr << "Unable to open file for writing!" << std::endl;
    }
  }
  return 0;
}
