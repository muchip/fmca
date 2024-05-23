
/* We test the triple product in FormattedMultiplication.h
We rely on the FMCA library by M.Multerer.
 */

#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/FormattedMultiplication"
#include "../FMCA/GradKernel"
#include "../FMCA/MatrixEvaluator"
#include "../FMCA/Samplets"
#include "../FMCA/SparseCompressor"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "read_files_txt.h"

#define DIM 2
#define GRADCOMPONENT

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> Sparse;

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
  // Initialize the points
  // int NPTS_SOURCE = 30000;
  // int NPTS_QUAD = 15000;
  // int N_WEIGHTS = NPTS_QUAD;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Vector w_vec;
  // P_sources = (FMCA::Matrix::Random(DIM, NPTS_SOURCE).array());
  // P_quad = (FMCA::Matrix::Random(DIM, NPTS_QUAD).array());
  // w_vec = 0.1 * FMCA::Vector::Random(NPTS_QUAD).array() + 1;
  readTXT("data/vertices_square500k.txt", P_sources, 2);
  readTXT("data/quadrature3_points_square500k.txt", P_quad, 2);
  readTXT("data/quadrature3_weights_square500k.txt", w_vec);
  int NPTS_SOURCE = P_sources.cols();
  int NPTS_QUAD = P_quad.cols();
  int N_WEIGHTS = NPTS_QUAD;
  //////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 4;
  const FMCA::Scalar threshold = 1e-2;
  const FMCA::Scalar MPOLE_DEG = 6;
  const FMCA::Scalar sigma = 0.05;
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const Moments mom_quad(P_quad, MPOLE_DEG);
  //////////////////////////////////////////////////////////////////////////////
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
  std::cout << "MPOLE_DEG = " << MPOLE_DEG << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // Create a sparse diagonal matrix from the vector 'w'
  FMCA::Vector w_perm = w_vec(permutationVector(hst_quad));
  FMCA::SparseMatrix<FMCA::Scalar> W(w_perm.size(), w_perm.size());
  for (int i = 0; i < w_perm.size(); ++i) {
    W.insert(i, i) = w_perm(i);
  }
  // Compress the weight matrix
  FMCA::SparseMatrixEvaluator mat_eval_weights(W);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp;
  Wcomp.init(hst_quad, eta, 1e-6);
  T.tic();
  Wcomp.compress(mat_eval_weights);
  double compressor_time_weights = T.toc();
  std::cout << "compression time weights:         " << compressor_time_weights
            << std::endl;
  const auto &trips_weights = Wcomp.triplets();
  std::cout << "anz:                                      "
            << trips_weights.size() / NPTS_QUAD << std::endl;
  Sparse Wcomp_Sparse(NPTS_QUAD, NPTS_QUAD);
  Wcomp_Sparse.setFromTriplets(trips_weights.begin(), trips_weights.end());
  Wcomp_Sparse.makeCompressed();
  //////////////////////////////////////////////////////////////////////////////
  for (int i = 0; i < DIM; ++i) {
    std::cout << std::string(80, '-') << std::endl;
    const FMCA::GradKernel function("MATERN32", sigma, 1, i);
    const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
    //////////////////////////////////////////////////////////////////////////////
    // gradKernel compression
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
    Scomp.init(hst_sources, hst_quad, eta, threshold);
    T.tic();
    Scomp.compress(mat_eval);
    double compressor_time = T.toc();
    std::cout << "compression time component " << i << ":         "
              << compressor_time << std::endl;
    const auto &trips = Scomp.triplets();
    std::cout << "anz:                                      "
              << trips.size() / NPTS_SOURCE << std::endl;
    Sparse Scomp_Sparse(NPTS_SOURCE, NPTS_QUAD);
    Scomp_Sparse.setFromTriplets(trips.begin(), trips.end());
    Scomp_Sparse.makeCompressed();
    //////////////////////////////////////////////////////////////////////////////
    // create the pattern for the triple product result --> the pattern is the
    // Samplet compression pattern
    const FMCA::CovarianceKernel kernel_funtion_ss("MATERN32", 0.1);
    const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources,
                                                   kernel_funtion_ss);
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
    Kcomp_ss.init(hst_sources, eta, threshold);
    std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets =
        Kcomp_ss.a_priori_pattern_triplets();
    Sparse pattern(NPTS_SOURCE, NPTS_SOURCE);
    pattern.setFromTriplets(a_priori_triplets.begin(), a_priori_triplets.end());
    pattern.makeCompressed();
    std::cout << "pattern size:        "
              << a_priori_triplets.size() / NPTS_SOURCE << std::endl;
    T.tic();
    formatted_sparse_multiplication_triple_product(
        pattern, Scomp_Sparse, Wcomp_Sparse.selfadjointView<Eigen::Upper>(),
        Scomp_Sparse);
    double mult_time = T.toc();
    std::cout << "multiplication time component " << i << ":             "
              << mult_time << std::endl;
    //////////////////////////////////////////////////////////////////////////////
    // Eigen multiplication
    T.tic();
    Sparse res_eigen = Scomp_Sparse *
                       Wcomp_Sparse.selfadjointView<Eigen::Upper>() *
                       Scomp_Sparse.transpose();
    double mult_time_eigen = T.toc();
    std::cout << "eigen multiplication time component " << i << ":      "
              << mult_time_eigen << std::endl;

    int num_entries = res_eigen.nonZeros();

    std::cout << "Number of entries in res_eigen: " << num_entries/NPTS_SOURCE << std::endl;
    return 0;
    //////////////////////////////////////////////////////////////////////////////
    // Comparison triple product vs eigen product
    // FMCA::Matrix pattern_matrix =
    // FMCA::Matrix(pattern).selfadjointView<Eigen::Upper>(); std::cout <<
    // "eigen-triple_prod:               "
    //           << (FMCA::Matrix(res_eigen) - pattern_matrix).norm() /
    //                  FMCA::Matrix(res_eigen).norm()
    //           << std::endl;
    Sparse pattern_full = pattern.selfadjointView<Eigen::Upper>();
    std::cout << "eigen-triple_prod row 1:               "
              << (res_eigen.row(0) - pattern_full.row(0)).norm() /
                     (res_eigen.row(0)).norm()
              << std::endl;
  }
  return 0;
}
