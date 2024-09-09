/* We test the triple product in FormattedMultiplication.h*/

#include <Eigen/Sparse>
#include <iostream>

#include "../FMCA/FormattedMultiplication"
#include "../FMCA/GradKernel"
#include "../FMCA/MatrixEvaluator"
#include "../FMCA/Samplets"
#include "../FMCA/SparseCompressor"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "FunctionsPDE.h"
#include "read_files_txt.h"

#define DIM 2

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> Sparse;
typedef Eigen::SparseMatrix<double> SparseEig;

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using SparseMatrixEvaluator = FMCA::SparseMatrixEvaluator;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using MatrixEvaluatorKernel =
    FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;

int main() {
  FMCA::Tictoc T;
  FMCA::Matrix P_sources_total;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Vector w_vec;
  FMCA::Matrix P_quad_border;
  FMCA::Vector w_vec_border;

  readTXT("data/Halton01_1M.txt", P_sources_total, DIM);
  P_sources = P_sources_total.leftCols(500000);
  readTXT("data/quadrature3_points_Unitsquare90k.txt", P_quad, DIM);
  readTXT("data/quadrature3_weights_Unitsquare90k.txt", w_vec);

  readTXT("data/quadrature01_border90k.txt", P_quad_border, DIM);
  readTXT("data/weights01_border90k.txt", w_vec_border);

  int NPTS_SOURCE = P_sources.cols();
  int NPTS_QUAD = P_quad.cols();
  int N_WEIGHTS = NPTS_QUAD;
  int NPTS_QUAD_BORDER = P_quad_border.cols();
  int N_WEIGHTS_BORDER = NPTS_QUAD_BORDER;

  std::cout << "sum weights:        " << w_vec.sum() << std::endl;
  std::cout << "sum weights border: " << w_vec_border.sum() << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 6;
  const FMCA::Scalar threshold = 1e-4;
  const FMCA::Scalar MPOLE_DEG = 10;
  const FMCA::Scalar sigma = 0.01;
  /////////////////////////////////////////////////////////////////////////////
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
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
  std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
  std::cout << "Number of weights = " << N_WEIGHTS << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "sigma = " << sigma << std::endl;
  std::cout << "eta = " << eta << std::endl;
  std::cout << "dtilde = " << dtilde << std::endl;
  std::cout << "threshold = " << threshold << std::endl;
  std::cout << "MPOLE_DEG = " << MPOLE_DEG << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  {
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Check interior integration" << std::endl;
  // Weights compression
  FMCA::SparseMatrix<FMCA::Scalar> W(w_vec.size(),
                                            w_vec.size());
  for (int i = 0; i < w_vec.size(); ++i) {
    W.insert(i, i) = w_vec(i);
  }
  const FMCA::SparseMatrixEvaluator mat_eval_weights(W);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp;
  std::cout << "Weights" << std::endl;
  Eigen::SparseMatrix<double> W_Sparse = createCompressedWeights(
      mat_eval_weights, hst_quad, eta, 0, N_WEIGHTS);

  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  const FMCA::CovarianceKernel function_kernel("MATERN32", sigma, 1);
  const usMatrixEvaluatorKernel mat_eval(mom_sources, mom_quad,
                                                function_kernel);
  //////////////////////////////////////////////////////////////////////////////
  // Kernel compression
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
  Scomp.init(hst_sources, hst_quad, eta, threshold);
  Scomp.compress(mat_eval);
  const auto &trips = Scomp.triplets();
  std::cout << "anz:                                      "
            << trips.size() / NPTS_SOURCE << std::endl;
  SparseEig Scomp_Sparse(NPTS_SOURCE, NPTS_QUAD);
  Scomp_Sparse.setFromTriplets(trips.begin(), trips.end());
  Scomp_Sparse.makeCompressed();
  //////////////////////////////////////////////////////////////////////////////
  // create the pattern for the triple product result --> the pattern is the
  // Samplet compression pattern
  const FMCA::CovarianceKernel kernel_funtion_ss("MATERN32", sigma);
  const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources,
                                                 kernel_funtion_ss);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
  Kcomp_ss.init(hst_sources, eta, threshold);
  std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets =
      Kcomp_ss.a_priori_pattern_triplets();
  Sparse pattern(NPTS_SOURCE, NPTS_SOURCE);
  pattern.setFromTriplets(a_priori_triplets.begin(),
                                 a_priori_triplets.end());
  pattern.makeCompressed();
  T.tic();

  formatted_sparse_multiplication_dotproduct(pattern, Scomp_Sparse,
                                             Scomp_Sparse);
  pattern *= w_vec(0);
  double mult_time = T.toc();
  std::cout << "multiplication time " << ":             " << mult_time
            << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  Sparse pattern_full = pattern.selfadjointView<Eigen::Upper>();
  FMCA::Vector v = FMCA::Vector::Random(NPTS_SOURCE).array();

  FMCA::Vector res1 = pattern_full * v;
  FMCA::Vector res2 = Scomp_Sparse.transpose() * v;
  FMCA::Vector res3 = Scomp_Sparse * res2;

  std::cout << "Error interior:                "
            << (res1 - w_vec(0) * res3).norm() << std::endl;
  }
  ///////////////////////////////////////////////////////////////////////////
  {
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Check border integration" << std::endl;
  // Weights compression
  FMCA::SparseMatrix<FMCA::Scalar> W_border(w_vec_border.size(),
                                            w_vec_border.size());
  for (int i = 0; i < w_vec_border.size(); ++i) {
    W_border.insert(i, i) = w_vec_border(i);
  }
  const FMCA::SparseMatrixEvaluator mat_eval_weights_border(W_border);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp_border;
  std::cout << "Weights_border" << std::endl;
  Eigen::SparseMatrix<double> W_Sparse_border = createCompressedWeights(
      mat_eval_weights_border, hst_quad_border, eta, 0, N_WEIGHTS_BORDER);

  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  const FMCA::CovarianceKernel function_kernel("MATERN32", sigma, 1);
  const usMatrixEvaluatorKernel mat_eval_border(mom_sources, mom_quad_border,
                                                function_kernel);
  //////////////////////////////////////////////////////////////////////////////
  // Kernel compression
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
  Kcomp.init(hst_sources, hst_quad_border, eta, threshold);
  T.tic();
  Kcomp.compress(mat_eval_border);
  double compressor_time = T.toc();
  const auto &trips_border = Kcomp.triplets();
  std::cout << "anz:                                      "
            << trips_border.size() / NPTS_SOURCE << std::endl;
  SparseEig Kcomp_Sparse(NPTS_SOURCE, NPTS_QUAD_BORDER);
  Kcomp_Sparse.setFromTriplets(trips_border.begin(), trips_border.end());
  Kcomp_Sparse.makeCompressed();
  //////////////////////////////////////////////////////////////////////////////
  // create the pattern for the triple product result --> the pattern is the
  // Samplet compression pattern
  const FMCA::CovarianceKernel kernel_funtion_ss("MATERN32", sigma);
  const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources,
                                                 kernel_funtion_ss);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
  Kcomp_ss.init(hst_sources, eta, threshold);
  std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets =
      Kcomp_ss.a_priori_pattern_triplets();
  Sparse pattern_border(NPTS_SOURCE, NPTS_SOURCE);
  pattern_border.setFromTriplets(a_priori_triplets.begin(),
                                 a_priori_triplets.end());
  pattern_border.makeCompressed();
  T.tic();
  // formatted_sparse_multiplication_triple_product(pattern_border,
  // Kcomp_Sparse,
  //                                                W_Sparse_border,
  //                                                Kcomp_Sparse);
  formatted_sparse_multiplication_dotproduct(pattern_border, Kcomp_Sparse,
                                             Kcomp_Sparse);
  pattern_border *= w_vec_border(0);
  double mult_time = T.toc();
  std::cout << "multiplication time " << ":             " << mult_time
            << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  Sparse pattern_full_border = pattern_border.selfadjointView<Eigen::Upper>();
  FMCA::Vector v = FMCA::Vector::Random(NPTS_SOURCE).array();

  FMCA::Vector res1 = pattern_full_border * v;
  FMCA::Vector res2 = Kcomp_Sparse.transpose() * v;
  FMCA::Vector res3 = Kcomp_Sparse * res2;

  std::cout << "Error border:                "
            << (res1 - w_vec_border(0) * res3).norm() << std::endl;
  }
  return 0;
}
