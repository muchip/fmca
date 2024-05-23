#include <cstdlib>
#include <iostream>
#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "NeumanNormalMultiplication.h"
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

double computeTrace(const Eigen::SparseMatrix<double> &mat) {
  double trace = 0.0;
  // Iterate only over the diagonal elements
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
      if (it.row() == it.col()) {  // Check if it is a diagonal element
        trace += it.value();
      }
    }
  }
  return trace;
}

int countSmallerThan(const Eigen::SparseMatrix<double> &matrix,
                     FMCA::Scalar threshold) {
  int count = 0;
  // Iterate over all non-zero elements.
  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it;
         ++it) {
      if (it.value() < threshold) {
        count++;
      }
    }
  }
  return count / matrix.rows();
}

bool isSymmetric(const Eigen::SparseMatrix<double> &matrix,
                 FMCA::Scalar tol = 1e-10) {
  if (matrix.rows() != matrix.cols())
    return false;  // Non-square matrices are not symmetric

  // Iterate over the outer dimension
  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it;
         ++it) {
      if (std::fabs(it.value() - matrix.coeff(it.col(), it.row())) > tol)
        return false;
    }
  }
  return true;
}

int main() {
  // POINTS
  FMCA::Tictoc T;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_quad_border;
  FMCA::Matrix Normals;
  FMCA::Vector w_vec;
  FMCA::Vector w_vec_border;
  // pointers
  readTXT("data/vertices_square_01.txt", P_sources, 2);
  readTXT("data/quadrature7_points_square_01.txt", P_quad, 2);
  readTXT("data/quadrature7_weights_square_01.txt", w_vec);
  readTXT("data/quadrature_border_01.txt", P_quad_border, 2);
  readTXT("data/weights_border_01.txt", w_vec_border);
  readTXT("data/normals_01.txt", Normals, 2);

  std::cout << "P_sources dim  " << P_sources.rows() << "," << P_sources.cols()
            << std::endl;
  std::cout << "P_quad dim  " << P_quad.rows() << "," << P_quad.cols()
            << std::endl;
  std::cout << "w_vec dim  " << w_vec.rows() << "," << w_vec.cols()
            << std::endl;
  std::cout << "P_quad_border dim  " << P_quad_border.rows() << ","
            << P_quad_border.cols() << std::endl;
  std::cout << "w_vec_border dim  " << w_vec_border.rows() << ","
            << w_vec_border.cols() << std::endl;
  std::cout << "Normals dim  " << Normals.rows() << "," << Normals.cols()
            << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar threshold_kernel = 1e-2;
  const FMCA::Scalar threshold_gradKernel = 1;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 4;
  const FMCA::Scalar sigma = 0.03;
  // const FMCA::Scalar sigma = 2 * 1 / (sqrt(P_sources.cols()));
  const FMCA::Scalar beta = 1000;
  std::string kernel_type = "EXPONENTIAL";
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const Moments mom_quad(P_quad, MPOLE_DEG);
  const Moments mom_quad_border(P_quad_border, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
  const SampletMoments samp_mom_quad_border(P_quad_border, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
  H2SampletTree hst_quad_border(mom_quad_border, samp_mom_quad_border, 0,
                                P_quad_border);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Number of source points =                         "
            << P_sources.cols() << std::endl;
  std::cout << "Number of quad points =                           "
            << P_quad.cols() << std::endl;
  std::cout << "Number of quad points border =                    "
            << P_quad_border.cols() << std::endl;
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
  std::cout << "threshold_kernel =                                "
            << threshold_kernel << std::endl;
  std::cout << "threshold_gradKernel =                            "
            << threshold_gradKernel << std::endl;
  std::cout << "threshold_weights =                               "
            << threshold_weights << std::endl;
  std::cout << "MPOLE_DEG =                                       " << MPOLE_DEG
            << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Kernel source-source compression for the pattern of the multiplication and
  // the final solution
  const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
  const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources,
                                                 kernel_funtion_ss);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
  Kcomp_ss.init(hst_sources, eta, threshold_kernel);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Weights compression
  FMCA::Vector w_perm = hst_quad.toClusterOrder(w_vec);
  FMCA::SparseMatrix<FMCA::Scalar> W(w_perm.size(), w_perm.size());
  for (int i = 0; i < w_perm.size(); ++i) {
    W.insert(i, i) = w_perm(i);
  }
  FMCA::SparseMatrixEvaluator mat_eval_weights(W);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp;
  Wcomp.init(hst_quad, eta, threshold_weights);
  T.tic();
  Wcomp.compress(mat_eval_weights);
  double compressor_time_weights = T.toc();
  const auto &trips_weights = Wcomp.triplets();
  std::cout << "anz weights:                                      "
            << trips_weights.size() / P_quad.cols() << std::endl;
  largeSparse Wcomp_Sparse(P_quad.cols(), P_quad.cols());
  Wcomp_Sparse.setFromTriplets(trips_weights.begin(), trips_weights.end());
  Wcomp_Sparse.makeCompressed();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Weights border compression
  FMCA::Vector w_perm_border = hst_quad_border.toClusterOrder(w_vec_border);
  FMCA::SparseMatrix<FMCA::Scalar> W_border(w_perm_border.size(),
                                            w_perm_border.size());
  for (int i = 0; i < w_perm_border.size(); ++i) {
    W_border.insert(i, i) = w_perm_border(i);
  }
  FMCA::SparseMatrixEvaluator mat_eval_weights_border(W_border);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp_border;
  Wcomp_border.init(hst_quad_border, eta, threshold_weights);
  Wcomp_border.compress(mat_eval_weights_border);
  const auto &trips_weights_border = Wcomp_border.triplets();
  std::cout << "anz weights border:                               "
            << trips_weights_border.size() / P_quad_border.cols() << std::endl;
  largeSparse Wcomp_border_Sparse(P_quad_border.cols(), P_quad_border.cols());
  Wcomp_border_Sparse.setFromTriplets(trips_weights_border.begin(),
                                      trips_weights_border.end());
  Wcomp_border_Sparse.makeCompressed();
  std::cout << std::string(80, '-') << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Kernel (sources,quad) compression
  const FMCA::CovarianceKernel kernel_funtion(kernel_type, sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel(mom_sources, mom_quad,
                                                kernel_funtion);
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
  Kcomp.init(hst_sources, hst_quad, eta, threshold_kernel);
  T.tic();
  Kcomp.compress(mat_eval_kernel);
  double compressor_time = T.toc();
  std::cout << "compression time kernel :                         "
            << compressor_time << std::endl;
  const auto &trips = Kcomp.triplets();
  std::cout << "anz kernel:                                       "
            << trips.size() / P_sources.cols() << std::endl;
  Eigen::SparseMatrix<double> Kcomp_Sparse(P_sources.cols(), P_quad.cols());
  Kcomp_Sparse.setFromTriplets(trips.begin(), trips.end());
  Kcomp_Sparse.makeCompressed();
  std::cout << std::string(80, '-') << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Penalty
  // construct the matrix penalty = K_{Psources, Pquad_border} * W_border *
  // K_{Psources, Pquad_border}.transpose()
  const FMCA::CovarianceKernel kernel_funtion_sources_quadborder(kernel_type,
                                                                 sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel_sources_quadborder(
      mom_sources, mom_quad_border, kernel_funtion_sources_quadborder);
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
      Kcomp_sources_quadborder;
  Kcomp_sources_quadborder.init(hst_sources, hst_quad_border, eta,
                                threshold_kernel);
  T.tic();
  Kcomp_sources_quadborder.compress(mat_eval_kernel_sources_quadborder);
  double compressor_time_sources_quadborder = T.toc();
  std::cout << "compression time kernel_sources_quadborder :      "
            << compressor_time_sources_quadborder << std::endl;
  const auto &trips_sources_quadborder = Kcomp_sources_quadborder.triplets();
  std::cout << "anz kernel_sources_quadborder :                         "
            << trips_sources_quadborder.size() / P_sources.cols() << std::endl;
  largeSparse Kcomp_sources_quadborder_Sparse(P_sources.cols(),
                                              P_quad_border.cols());
  Kcomp_sources_quadborder_Sparse.setFromTriplets(
      trips_sources_quadborder.begin(), trips_sources_quadborder.end());
  Kcomp_sources_quadborder_Sparse.makeCompressed();
  std::cout << std::string(80, '-') << std::endl;
  // M_b
  std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets_penalty =
      Kcomp_ss.a_priori_pattern_triplets();
  largeSparse pattern_penalty(P_sources.cols(), P_sources.cols());
  pattern_penalty.setFromTriplets(a_priori_triplets_penalty.begin(),
                                  a_priori_triplets_penalty.end());
  pattern_penalty.makeCompressed();
  std::cout<< a_priori_triplets_penalty.size() << std::endl;

  T.tic();
  formatted_sparse_multiplication_triple_product(
      pattern_penalty, Kcomp_sources_quadborder_Sparse,
      Wcomp_border_Sparse.selfadjointView<Eigen::Upper>(),
      Kcomp_sources_quadborder_Sparse);
  double mult_time_penalty = T.toc();
  std::cout << "multiplication time penalty:              " << mult_time_penalty
            << std::endl;
  T.tic();
  largeSparse Mb = Kcomp_sources_quadborder_Sparse *
                   (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
                    Kcomp_sources_quadborder_Sparse.transpose())
                       .eval();
  double mult_time_eigen_penalty = T.toc();
  std::cout << "multiplication eigen time penalty:              "
            << mult_time_eigen_penalty << std::endl;
  // Comparison triple product vs eigen product
  FMCA::Matrix pattern_matrix =
      FMCA::Matrix(pattern_penalty).selfadjointView<Eigen::Upper>();
  FMCA::Matrix Mb_matrix = FMCA::Matrix(Mb);
  std::cout << "eigen-triple_prod:               "
            << (Mb_matrix - pattern_matrix).norm() /
                   Mb_matrix.norm()
            << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // create stiffness and Neumann term
  largeSparse stiffness(P_sources.cols(), P_sources.cols());
  stiffness.setZero();

  for (FMCA::Index i = 0; i < DIM; ++i) {
    std::cout << std::string(80, '-') << std::endl;
    const FMCA::GradKernel function(kernel_type, sigma, 1, i);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// stiffness
    const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
    Scomp.init(hst_sources, hst_quad, eta, threshold_gradKernel);
    T.tic();
    Scomp.compress(mat_eval);
    double compressor_time = T.toc();
    std::cout << "compression time gradKernel " << i << "                      "
              << compressor_time << std::endl;
    const auto &trips = Scomp.triplets();
    std::cout << "anz gradKernel:                                   "
              << trips.size() / P_sources.cols() << std::endl;
    Eigen::SparseMatrix<double> Scomp_Sparse(P_sources.cols(), P_quad.cols());
    Scomp_Sparse.setFromTriplets(trips.begin(), trips.end());
    Scomp_Sparse.makeCompressed();

    std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets_grad =
    Kcomp_ss.a_priori_pattern_triplets();
    largeSparse pattern_grad(P_sources.cols(), P_sources.cols());
    pattern_grad.setFromTriplets(a_priori_triplets_grad.begin(),
                                  a_priori_triplets_grad.end());
    pattern_grad.makeCompressed();
    std::cout<< a_priori_triplets_grad.size() << std::endl;

    largeSparse Wcomp_Sparse_full = Wcomp_Sparse.selfadjointView<Eigen::Upper>();
    T.tic();
    formatted_sparse_multiplication_triple_product(pattern_grad, Scomp_Sparse, Wcomp_Sparse_full, Scomp_Sparse);
    double mult_time = T.toc();
    std::cout << "mult time component " << i << "                        "
              << mult_time << std::endl;

    T.tic();
    largeSparse gradk = Scomp_Sparse *
                   (Wcomp_Sparse_full * Scomp_Sparse.transpose()).eval();
  double mult_time_eigen_grad = T.toc();
  std::cout << "multiplication eigen time grad:              "
            << mult_time_eigen_grad << std::endl;
  // Comparison triple product vs eigen product
  FMCA::Matrix pattern_matrix_grad =
      FMCA::Matrix(pattern_grad).selfadjointView<Eigen::Upper>();
  FMCA::Matrix grad_matrix = FMCA::Matrix(gradk);
  std::cout << "eigen-triple_prod:               "
            << (grad_matrix - pattern_matrix_grad).norm() /
                   grad_matrix.norm()
            << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  return 0;
}
