#include <algorithm>
#include <cstdlib>
#include <iostream>
// ##############################
#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/Samplets/adaptiveTreeSearch.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "FunctionsPDE.h"
#include "read_files_txt.h"

#define DIM 2

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
using EigenCholesky =
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>;
using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
  // POINTS
  FMCA::Tictoc T;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Vector w_vec;
  FMCA::Matrix P_quad_test;
  FMCA::Vector w_vec_test;
  readTXT("data/vertices_square2500.txt", P_sources, 2);
  readTXT("data/quadrature3_points_square2500.txt", P_quad, 2);
  readTXT("data/quadrature3_weights_square2500.txt", w_vec);
  readTXT("data/quadrature3_points_square2k.txt", P_quad_test, 2);
  readTXT("data/quadrature3_weights_square2k.txt", w_vec_test);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 1./ DIM;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 4;
  const std::string kernel_type = "MATERN32";
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // FMCA::Scalar threshold_gradKernel = threshold_kernel / (sigma * sigma);
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const Moments mom_quad(P_quad, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
  const Moments mom_quad_test(P_quad_test, MPOLE_DEG);
  const SampletMoments samp_mom_quad_test(P_quad_test, dtilde - 1);
  H2SampletTree hst_quad_test(mom_quad_test, samp_mom_quad_test, 0,
                              P_quad_test);

  std::vector<FMCA::Scalar> sigmas = {0.03};
  for (FMCA::Scalar sigma : sigmas) {
    FMCA::Scalar threshold_gradKernel = 1e-4; // you can relate it to sigma 
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Number of source points =                         "
              << P_sources.cols() << std::endl;
    std::cout << "Number of quad points =                           "
              << P_quad.cols() << std::endl;
    std::cout << "Number of quad points test=                       "
              << P_quad_test.cols() << std::endl;
    std::cout << "minimum element:                                  "
              << *std::min_element(w_vec.begin(), w_vec.end()) << std::endl;
    std::cout << "maximum element:                                  "
              << *std::max_element(w_vec.begin(), w_vec.end()) << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "sigma =                                           " << sigma
              << std::endl;
    std::cout << "eta =                                             " << eta
              << std::endl;
    std::cout << "dtilde =                                          " << dtilde
              << std::endl;
    std::cout << "threshold_gradKernel =                            "
              << threshold_gradKernel << std::endl;
    std::cout << "threshold_weights =                               "
              << threshold_weights << std::endl;
    std::cout << "MPOLE_DEG =                                       "
              << MPOLE_DEG << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Weights compression
    FMCA::Vector w_perm = hst_quad.toClusterOrder(w_vec);
    FMCA::SparseMatrix<FMCA::Scalar> W(w_perm.size(), w_perm.size());
    for (int i = 0; i < w_perm.size(); ++i) {
      W.insert(i, i) = w_perm(i);
    }
    const FMCA::SparseMatrixEvaluator mat_eval_weights(W);
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp;
    std::cout << "Weights" << std::endl;
    Eigen::SparseMatrix<double> Wcomp_Sparse = createCompressedWeights(
        mat_eval_weights, hst_quad, eta, threshold_weights, P_quad.cols());

    Eigen::SparseMatrix<double> Wcomp_upperlower =
        (Wcomp_Sparse.selfadjointView<Eigen::Upper>());
    Eigen::MatrixXd Wcomp_upperlower_dense = Wcomp_upperlower.toDense();

    FMCA::Matrix W_full;
    mat_eval_weights.compute_dense_block(hst_quad, hst_quad, &W_full);
    W_full = hst_quad.sampletTransform(W_full);
    W_full = hst_quad.sampletTransform(W_full.transpose()).transpose();
    FMCA::Matrix difference = Wcomp_upperlower_dense - W_full;
    std::cout << "weights compressed - full weights :  "
              << (Wcomp_upperlower_dense - W_full).norm() / W_full.norm()
              << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Weights test compression
    FMCA::Vector w_perm_test = hst_quad_test.toClusterOrder(w_vec_test);
    FMCA::SparseMatrix<FMCA::Scalar> W_test(w_perm_test.size(),
                                            w_perm_test.size());
    for (int i = 0; i < w_perm_test.size(); ++i) {
      W_test.insert(i, i) = w_perm_test(i);
    }
    const FMCA::SparseMatrixEvaluator mat_eval_weights_test(W_test);
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp_test;
    std::cout << "Weights test" << std::endl;
    Eigen::SparseMatrix<double> Wcomp_Sparse_test =
        createCompressedWeights(mat_eval_weights_test, hst_quad_test, eta,
                                threshold_weights, P_quad_test.cols());

    Eigen::SparseMatrix<double> Wcomp_upperlower_test =
        (Wcomp_Sparse_test.selfadjointView<Eigen::Upper>());
    Eigen::MatrixXd Wcomp_upperlower_dense_test =
        Wcomp_upperlower_test.toDense();

    FMCA::Matrix W_full_test;
    mat_eval_weights_test.compute_dense_block(hst_quad_test, hst_quad_test,
                                              &W_full_test);
    W_full_test = hst_quad_test.sampletTransform(W_full_test);
    W_full_test =
        hst_quad_test.sampletTransform(W_full_test.transpose()).transpose();
    FMCA::Matrix difference_test = Wcomp_upperlower_dense_test - W_full_test;
    std::cout << "weights compressed_test - full weights_test :  "
              << (Wcomp_upperlower_dense - W_full).norm() / W_full.norm()
              << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// stiffness
    FMCA::Matrix stiffness_full(P_sources.cols(), P_sources.cols());
    Eigen::SparseMatrix<double> stiffness(P_sources.cols(), P_sources.cols());
    stiffness.setZero();
    FMCA::Scalar anz_gradkernel = 0;
    for (FMCA::Index i = 0; i < DIM; ++i) {
      std::cout << std::string(80, '-') << std::endl;
      const FMCA::GradKernel function(kernel_type, sigma, 1, i);
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
      FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
      std::cout << "GradK" << std::endl;
      Eigen::SparseMatrix<double> Scomp_Sparse =
          createCompressedSparseMatrixUnSymmetric(
              function, mat_eval, hst_sources, hst_quad, eta,
              threshold_gradKernel, P_sources, P_quad);

      FMCA::Matrix grad_full;
      mat_eval.compute_dense_block(hst_sources, hst_quad, &grad_full);
      grad_full = hst_sources.sampletTransform(grad_full);
      grad_full = hst_quad.sampletTransform(grad_full.transpose()).transpose();
      FMCA::Matrix difference = Scomp_Sparse.toDense() - grad_full;
      std::cout << "grad compressed - grad weights :  "
                << (Scomp_Sparse.toDense() - grad_full).norm() /
                       grad_full.norm()
                << std::endl;

      stiffness_full += grad_full * W_full * grad_full.transpose();

      Eigen::SparseMatrix<double> gradk =
          Scomp_Sparse * (Wcomp_Sparse * Scomp_Sparse.transpose()).eval();
      // applyPattern(gradk, triplets_pattern);
      gradk.makeCompressed();
      stiffness += gradk.selfadjointView<Eigen::Upper>();
    }

    std::cout << "stiffness compressed - full stiffness :  "
              << (stiffness.toDense() - stiffness_full).norm() /
                     stiffness_full.norm()
              << std::endl;

    FMCA::Matrix stiffness_full_test(P_sources.cols(), P_sources.cols());
    Eigen::SparseMatrix<double> stiffness_test(P_sources.cols(),
                                               P_sources.cols());
    stiffness_test.setZero();
    for (FMCA::Index i = 0; i < DIM; ++i) {
      std::cout << std::string(80, '-') << std::endl;
      const FMCA::GradKernel function_test(kernel_type, sigma, 1, i);
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /// stiffness
      const usMatrixEvaluator mat_eval_test(mom_sources, mom_quad_test,
                                            function_test);
      FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
          Scomp_test;
      std::cout << "GradK_test" << std::endl;
      Eigen::SparseMatrix<double> Scomp_Sparse_test =
          createCompressedSparseMatrixUnSymmetric(
              function_test, mat_eval_test, hst_sources, hst_quad_test, eta,
              threshold_gradKernel, P_sources, P_quad_test);

      FMCA::Matrix grad_full_test;
      mat_eval_test.compute_dense_block(hst_sources, hst_quad_test,
                                        &grad_full_test);
      grad_full_test = hst_sources.sampletTransform(grad_full_test);
      grad_full_test =
          hst_quad_test.sampletTransform(grad_full_test.transpose())
              .transpose();
      FMCA::Matrix difference_test =
          Scomp_Sparse_test.toDense() - grad_full_test;
      std::cout << "grad compressed_test - grad weights_test :  "
                << (Scomp_Sparse_test.toDense() - grad_full_test).norm() /
                       grad_full_test.norm()
                << std::endl;

      stiffness_full_test +=
          grad_full_test * W_full_test * grad_full_test.transpose();

      Eigen::SparseMatrix<double> gradk_test =
          Scomp_Sparse_test *
          (Wcomp_Sparse_test * Scomp_Sparse_test.transpose()).eval();
      // applyPattern(gradk, triplets_pattern);
      gradk_test.makeCompressed();
      stiffness_test += gradk_test.selfadjointView<Eigen::Upper>();
    }

    std::cout << "stiffness compressed_test - full stiffness_test :  "
              << (stiffness_test.toDense() - stiffness_full_test).norm() /
                     stiffness_full_test.norm()
              << std::endl;

    std::cout << "stiffness compressed_test - full stiffness  :  "
              << (stiffness.toDense() - stiffness_test.toDense()).norm() /
                     stiffness.norm()
              << std::endl;
  }
  return 0;
}
