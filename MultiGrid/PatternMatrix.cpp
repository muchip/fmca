#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

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


using namespace FMCA;

void runForMu(Scalar nu) {
  Tictoc T;
  ///////////////////////////////// Inputs: points + maximum level
  Matrix P1, P2, P3;
  readTXT("data/square01_uniform_grid_level6.txt", P1, DIM);
  readTXT("data/square01_uniform_grid_level8.txt", P2, DIM);
  readTXT("data/square01_uniform_grid_level10.txt", P3, DIM);

  ///////////////////////////////// Nested cardinality of points
  std::vector<Matrix> P_Matrices = {P1, P2, P3};
  int max_level = P_Matrices.size();
  ///////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar threshold_kernel = 1e-6;
  const Scalar threshold_aPost = -1;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = 2 * (dtilde - 1);
  const std::string kernel_type = "wendland21";
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "threshold_kernel    " << threshold_kernel << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;
  std::cout << "kernel_type         " << kernel_type << std::endl;
  std::cout << "nu                  " << nu << std::endl;

  ///////////////////////////////// Rhs
  Vector fill_distances(max_level);
  for (Index i = 0; i < max_level; ++i) {
    const Moments mom(P_Matrices[i], mpole_deg);
    const SampletMoments samp_mom(P_Matrices[i], dtilde - 1);
    const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P_Matrices[i]);
    Vector minDistance = minDistanceVector(hst, P_Matrices[i]);
    Scalar h = minDistance.maxCoeff();
    fill_distances[i] = h;
  }

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
      std::string name = "A_" + std::to_string(l) + "_" + std::to_string(j);
      {
        Eigen::SparseMatrix<double> B_comp = UnsymmetricCompressor(
            mom, samp_mom, hst, kernel_funtion_B, eta, threshold_kernel,
            threshold_aPost, mpole_deg, dtilde, P_Matrices[l], P_Matrices[j]);
        FMCA::IO::print2spascii(name, B_comp, "w");
      }  // B_comp goes out of scope and is destroyed here
      std::cout << "------------------------------------------" << std::endl;
    }

    Scalar sigma = nu * fill_distances[l];
    const CovarianceKernel kernel_funtion(kernel_type, sigma);
    std::string name_diag = "A_" + std::to_string(l) + "_" + std::to_string(l);
    {
      Eigen::SparseMatrix<double> A_comp = SymmetricCompressor(
          mom, samp_mom, hst, kernel_funtion, eta, threshold_kernel,
          threshold_aPost, mpole_deg, dtilde, P_Matrices[l]);
          FMCA::IO::print2spascii(name_diag, A_comp, "w");
      std::cout << "------------------------------------------" << std::endl;
    }  // A_comp goes out of scope and is destroyed here
  }
}

int main() {
  // std::vector<Scalar> nus = {0.5, 1, 1.5, 2};
  std::vector<Scalar> nus = {4};
  for (Scalar nu : nus) {
    runForMu(nu);
  }
  return 0;
}
