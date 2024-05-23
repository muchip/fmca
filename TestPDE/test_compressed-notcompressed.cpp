#include <cstdlib>
#include <iostream>
#include <fstream>

//
#include </opt/homebrew/Cellar/metis/5.1.0/include/metis.h>

#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/MetisSupport>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Sparse>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/SparseCholesky>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/SparseQR>
#include <Eigen/OrderingMethods>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "Uzawa.h"
#include "read_files_txt.h"

#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluatorKernel =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using EigenCholesky =
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                         Eigen::MetisOrdering<int>>;

int main() {
  // Initialize the matrices of source points, quadrature points, weights
  FMCA::Tictoc T;
  int NPTS_SOURCE;
  int NPTS_SOURCE_BORDER;
  int NPTS_QUAD;
  int NPTS_QUAD_BORDER;
  int N_WEIGHTS;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_quad_border;
  FMCA::Vector w_vec;
  //////////////////////////////////////////////////////////////////////////////
  readTXT("data/vertices_square_01.txt", P_sources, 2);
  readTXT("data/quadrature7_points_square_01.txt", P_quad, 2);
  readTXT("data/quadrature7_weights_square_01.txt", w_vec);
    // P_sources = (FMCA::Matrix::Random(DIM, 10000).array());
    // P_quad = (FMCA::Matrix::Random(DIM, 10000).array());
    // w_vec = 0.002 * FMCA::Vector::Random(10000).array() + 0.004;
//////////////////////////////////////////////////////////////////////
NPTS_SOURCE = P_sources.cols();
NPTS_QUAD = P_quad.cols();
N_WEIGHTS = w_vec.rows();
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 4;
  const FMCA::Scalar threshold = 1e-6;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 6;
  const FMCA::Scalar sigma = 0.1;
  //////////////////////////////////////////////////////////////////////////////
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const Moments mom_quad(P_quad, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
  std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
  std::cout << "Number of weights = " << N_WEIGHTS << std::endl;
  std::cout << "minimum element:                     "
            << *std::min_element(w_vec.begin(), w_vec.end()) << std::endl;
  std::cout << "maximum element:                     "
            << *std::max_element(w_vec.begin(), w_vec.end()) << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "sigma = " << sigma << std::endl;
  std::cout << "eta = " << eta << std::endl;
  std::cout << "dtilde = " << dtilde << std::endl;
  std::cout << "threshold = " << threshold << std::endl;
  std::cout << "threshold_weights = " << threshold_weights << std::endl;
  std::cout << "MPOLE_DEG = " << MPOLE_DEG << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // Weights matrix compressed
  FMCA::Vector w_perm = w_vec(permutationVector(hst_quad));
  FMCA::SparseMatrix<FMCA::Scalar> W(w_perm.size(), w_perm.size());
  for (int i = 0; i < w_perm.size(); ++i) {
    W.insert(i, i) = w_perm(i);
  }

  FMCA::SparseMatrixEvaluator mat_eval_weights(W);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp;
  Wcomp.init(hst_quad, eta, threshold_weights);
  Wcomp.compress(mat_eval_weights);
  const auto &trips_weights = Wcomp.triplets();
  Eigen::SparseMatrix<double> Wcomp_Sparse(NPTS_QUAD, NPTS_QUAD);

  Wcomp_Sparse.setFromTriplets(trips_weights.begin(), trips_weights.end());
  Wcomp_Sparse.makeCompressed();

  Eigen::SparseMatrix<double> Wcomp_upperlower = (Wcomp_Sparse.selfadjointView<Eigen::Upper>());
  Eigen::MatrixXd Wcomp_upperlower_dense = Wcomp_upperlower.toDense();

  FMCA::Matrix W_full;
  mat_eval_weights.compute_dense_block(hst_quad, hst_quad, &W_full);
  W_full = hst_quad.sampletTransform(W_full);
  W_full = hst_quad.sampletTransform(W_full.transpose()).transpose();
  FMCA::Matrix difference = Wcomp_upperlower_dense - W_full;
  std::cout << "weights compressed - full weights :  "
            << (Wcomp_upperlower_dense - W_full).norm() / W_full.norm()
            << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  const FMCA::CovarianceKernel kernel_funtion("MATERN32", sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel(mom_sources, mom_quad,
                                                kernel_funtion);
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
  Kcomp.init(hst_sources, hst_quad, eta, threshold);
  Kcomp.compress(mat_eval_kernel);
  const auto &trips = Kcomp.triplets();
  Eigen::SparseMatrix<double> Kcomp_Sparse(NPTS_SOURCE, NPTS_QUAD);
  Kcomp_Sparse.setFromTriplets(trips.begin(), trips.end());
  Kcomp_Sparse.makeCompressed();

  FMCA::Matrix K_full;
  mat_eval_kernel.compute_dense_block(hst_sources, hst_quad, &K_full);
  K_full = hst_sources.sampletTransform(K_full);
  K_full = hst_quad.sampletTransform(K_full.transpose()).transpose();

  std::cout << "kernel compressed - full kernel :  "
            << (Kcomp_Sparse.toDense() - K_full).norm() / K_full.norm()
            << std::endl;

  // Eigen::SparseMatrix<double> mass =
  //     Kcomp_Sparse *
  //     (W_nc.sparseView() * Kcomp_Sparse.transpose())
  //         .eval();

  // //////////////////////////////////////////////////////////////////////////////
  // Eigen::SparseMatrix<double> stiffness(NPTS_SOURCE, NPTS_SOURCE);
  // stiffness.setZero();

  for (int i = 0; i < DIM; ++i) {
    std::cout << std::string(80, '-') << std::endl;
    const FMCA::GradKernel function("MATERN32", sigma, 1, i);
    const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
    //////////////////////////////////////////////////////////////////////////////
    // gradKernel compression
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
    Scomp.init(hst_sources, hst_quad, eta, threshold);
    Scomp.compress(mat_eval);
    const auto &trips = Scomp.triplets();
    Eigen::SparseMatrix<double> Scomp_Sparse(NPTS_SOURCE, NPTS_QUAD);
    Scomp_Sparse.setFromTriplets(trips.begin(), trips.end());
    Scomp_Sparse.makeCompressed();

    FMCA::Matrix gradK_full;
    mat_eval.compute_dense_block(hst_sources, hst_quad, &gradK_full);
    gradK_full = hst_sources.sampletTransform(gradK_full);
    gradK_full = hst_quad.sampletTransform(gradK_full.transpose()).transpose();

    std::cout << "grad kernel compressed - full grad kernel component " << i << ":    "
              << (Scomp_Sparse.toDense() - gradK_full).norm() / gradK_full.norm()
              << std::endl;
  }
    return 0;
  }
