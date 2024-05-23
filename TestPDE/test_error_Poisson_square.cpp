#include <iostream>
#include <cstdlib>
#include <algorithm>
// ##############################
#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

#include "../FMCA/GradKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/Samplets/adaptiveTreeSearch.h"
#include "../FMCA/src/util/Grid3D.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "FunctionsPDE.h"
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
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using EigenCholesky =
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>;
using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FMCA::Scalar ComputeError(FMCA::Matrix &P_sources) {
  // POINTS
  FMCA::Tictoc T;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_quad_border;
  FMCA::Matrix Normals;
  FMCA::Vector w_vec;
  FMCA::Vector w_vec_border;
  //   readTXT("data/vertices_square_Halton500.txt", P_sources, 2);
  readTXT("data/quadrature3_points_square40k.txt", P_quad, 2);
  readTXT("data/quadrature3_weights_square40k.txt", w_vec);
  readTXT("data/quadrature_border40k.txt", P_quad_border, 2);
  readTXT("data/weights_border40k.txt", w_vec_border);
  readTXT("data/normals40k.txt", Normals, 2);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // U_BC vector
  FMCA::Vector u_bc(P_quad_border.cols());
  for (FMCA::Index i = 0; i < P_quad_border.cols(); ++i) {
    FMCA::Scalar x = P_quad_border(0, i);
    FMCA::Scalar y = P_quad_border(1, i);
    u_bc[i] = 0;
  }
  // Right hand side
  FMCA::Vector f(P_quad.cols());
  for (FMCA::Index i = 0; i < P_quad.cols(); ++i) {
    FMCA::Scalar x = P_quad(0, i);
    FMCA::Scalar y = P_quad(1, i);
    f[i] = 2 * FMCA_PI * FMCA_PI * sin(FMCA_PI * x) * sin(FMCA_PI * y);
    // f[i] = -2;
  }
  // Analytical sol of the problem
  FMCA::Vector analytical_sol(P_sources.cols());
  for (int i = 0; i < P_sources.cols(); ++i) {
    double x = P_sources(0, i);
    double y = P_sources(1, i);
    analytical_sol[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar threshold_kernel = 1e-6;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 4;
  const FMCA::Scalar beta = 10000;
  const std::string kernel_type = "MATERN32";
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // FMCA::Scalar threshold_gradKernel = threshold_kernel / (sigma * sigma);
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

  //   FMCA::Vector minDistance = minDistanceVector(hst_sources, P_sources);
  //   auto maxElementIterator =
  //       std::max_element(minDistance.begin(), minDistance.end());
  //   FMCA::Scalar sigma_h = *maxElementIterator;
  //   std::cout << "fill distance:                      " << sigma_h <<
  //   std::endl;
  FMCA::Scalar sigma = 0.1;
  FMCA::Scalar threshold_gradKernel = 1e-2;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Kernel source-source compression
  const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
  const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources,
                                                 kernel_funtion_ss);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
  Kcomp_ss.init(hst_sources, eta, threshold_kernel);
  std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets =
      Kcomp_ss.a_priori_pattern_triplets();
  Kcomp_ss.compress(mat_eval_kernel_ss);
  const auto &triplets = Kcomp_ss.triplets();
  int anz = triplets.size() / P_sources.cols();

  Eigen::SparseMatrix<double> Kcomp_ss_Sparse(P_sources.cols(),
                                              P_sources.cols());
  Kcomp_ss_Sparse.setFromTriplets(triplets.begin(), triplets.end());
  Kcomp_ss_Sparse.makeCompressed();
  // std::vector<Eigen::Triplet<FMCA::Scalar>> triplets_pattern =
  // TripletsPatternSymmetric(P_sources.cols(), hst_sources, eta,
  // threshold_kernel);
  //   Eigen::SparseMatrix<double> Kcomp_ss_Sparse =
  //   createCompressedSparseMatrixSymmetric(
  //       kernel_funtion_ss, mat_eval_kernel_ss, hst_sources, eta,
  //       threshold_kernel, P_sources.cols());
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Weights compression
  FMCA::Vector w_perm = hst_quad.toClusterOrder(w_vec);
  FMCA::SparseMatrix<FMCA::Scalar> W(w_perm.size(), w_perm.size());
  for (int i = 0; i < w_perm.size(); ++i) {
    W.insert(i, i) = w_perm(i);
  }
  const FMCA::SparseMatrixEvaluator mat_eval_weights(W);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp;
  Eigen::SparseMatrix<double> Wcomp_Sparse = createCompressedWeights(
      mat_eval_weights, hst_quad, eta, threshold_weights, P_quad.cols());
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Weights border compression
  FMCA::Vector w_perm_border = hst_quad_border.toClusterOrder(w_vec_border);
  FMCA::SparseMatrix<FMCA::Scalar> W_border(w_perm_border.size(),
                                            w_perm_border.size());
  for (int i = 0; i < w_perm_border.size(); ++i) {
    W_border.insert(i, i) = w_perm_border(i);
  }
  FMCA::SparseMatrixEvaluator mat_eval_weights_border(W_border);
  Eigen::SparseMatrix<double> Wcomp_border_Sparse =
      createCompressedWeights(mat_eval_weights_border, hst_quad_border, eta,
                              threshold_weights, P_quad_border.cols());
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Kernel (sources,quad) compression
  const FMCA::CovarianceKernel kernel_funtion(kernel_type, sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel(mom_sources, mom_quad,
                                                kernel_funtion);
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
  Eigen::SparseMatrix<double> Kcomp_Sparse =
      createCompressedSparseMatrixUnSymmetric(
          kernel_funtion, mat_eval_kernel, hst_sources, hst_quad, eta,
          threshold_kernel, P_sources.cols(), P_quad.cols());
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Penalty: construct the matrix M_b = K_{Psources, Pquad_border} * W_border
  // * K_{Psources, Pquad_border}.transpose()
  const FMCA::CovarianceKernel kernel_funtion_sources_quadborder(kernel_type,
                                                                 sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel_sources_quadborder(
      mom_sources, mom_quad_border, kernel_funtion_sources_quadborder);
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
      Kcomp_sources_quadborder;
  Eigen::SparseMatrix<double> Kcomp_sources_quadborder_Sparse =
      createCompressedSparseMatrixUnSymmetric(
          kernel_funtion_sources_quadborder, mat_eval_kernel_sources_quadborder,
          hst_sources, hst_quad_border, eta, threshold_kernel, P_sources.cols(),
          P_quad_border.cols());
  // M_b
  // std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets_penalty =
  //     Kcomp_ss.a_priori_pattern_triplets();
  // Eigen::SparseMatrix<double> pattern_penalty(P_sources.cols(),
  // P_sources.cols());
  // pattern_penalty.setFromTriplets(a_priori_triplets_penalty.begin(),
  // a_priori_triplets_penalty.end()); pattern_penalty.makeCompressed();
  T.tic();
  // formatted_sparse_multiplication_triple_product(pattern_penalty,
  // Kcomp_sources_quadborder_Sparse,
  // Wcomp_border_Sparse.selfadjointView<Eigen::Upper>(),
  // Kcomp_sources_quadborder_Sparse);
  Eigen::SparseMatrix<double> Mb =
      Kcomp_sources_quadborder_Sparse *
      (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
       Kcomp_sources_quadborder_Sparse.transpose())
          .eval();
  // applyPattern(Mb, triplets_pattern);
  double mult_time_eigen_penalty = T.toc();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // create stiffness and Neumann term
  Eigen::SparseMatrix<double> stiffness(P_sources.cols(), P_sources.cols());
  stiffness.setZero();

  Eigen::SparseMatrix<double> grad_times_normal(P_sources.cols(),
                                                P_quad_border.cols());
  grad_times_normal.setZero();
  FMCA::Scalar anz_gradkernel = 0;
  for (FMCA::Index i = 0; i < DIM; ++i) {
    const FMCA::GradKernel function(kernel_type, sigma, 1, i);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// stiffness
    const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
    Eigen::SparseMatrix<double> Scomp_Sparse =
        createCompressedSparseMatrixUnSymmetric(
            function, mat_eval, hst_sources, hst_quad, eta,
            threshold_gradKernel, P_sources.cols(), P_quad.cols());

    T.tic();
    Sparse pattern(P_sources.cols(), P_sources.cols());
    pattern.setFromTriplets(a_priori_triplets.begin(), a_priori_triplets.end());
    pattern.makeCompressed();
    //   Eigen::SparseMatrix<double> gradk =
    //       Scomp_Sparse * (Wcomp_Sparse * Scomp_Sparse.transpose()).eval();
    formatted_sparse_multiplication_dotproduct(pattern, Scomp_Sparse,
                                               Scomp_Sparse);
    //   Eigen::SparseMatrix<double> gradk =
    //       Scomp_Sparse * Scomp_Sparse.transpose();
    pattern *= w_vec(0);
    // applyPattern(gradk, triplets_pattern);
    double mult_time_eigen = T.toc();
    pattern.makeCompressed();
    stiffness += pattern.selfadjointView<Eigen::Upper>();
    // stiffness += gradk;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Neumann
    const usMatrixEvaluator mat_eval_neumann(mom_sources, mom_quad_border,
                                             function);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
        Scomp_neumann;
    Eigen::SparseMatrix<double> Scomp_Sparse_neumann =
        createCompressedSparseMatrixUnSymmetric(
            function, mat_eval_neumann, hst_sources, hst_quad_border, eta,
            threshold_gradKernel, P_sources.cols(), P_quad_border.cols());

    Eigen::SparseMatrix<double> gradk_n =
        NeumannNormalMultiplication(Scomp_Sparse_neumann, Normals.row(i));
    gradk_n.makeCompressed();
    grad_times_normal += gradk_n;
  }
  anz_gradkernel = anz_gradkernel / DIM;

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Final Neumann Term
  Eigen::SparseMatrix<double> Neumann =
      grad_times_normal * (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
                           Kcomp_sources_quadborder_Sparse.transpose())
                              .eval();
  // Nitsche’s Term
  Eigen::SparseMatrix<double> Neumann_Nitsche = Neumann.transpose();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // reorder the boundary conditions and the rhs to follow the samplets order
  FMCA::Vector u_bc_reordered = hst_quad_border.toClusterOrder(u_bc);
  FMCA::Vector f_reordered = hst_quad.toClusterOrder(f);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // F_comp = right hand side of the problem involving the source term f
  FMCA::Vector F_comp =
      Kcomp_Sparse * (Wcomp_Sparse.selfadjointView<Eigen::Upper>() *
                      hst_quad.sampletTransform(f_reordered))
                         .eval();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // G_comp = right hand side penalty
  FMCA::Vector G_comp = Kcomp_sources_quadborder_Sparse *
                        (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
                         hst_quad_border.sampletTransform(u_bc_reordered))
                            .eval();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // N_comp = right hand side Nitsche
  FMCA::Vector N_comp =
      grad_times_normal * (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
                           hst_quad_border.sampletTransform(u_bc_reordered))
                              .eval();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::SparseMatrix<double> Matrix_system_half =
      (stiffness + beta * Mb - (Neumann + Neumann_Nitsche));
  applyPattern(Matrix_system_half, a_priori_triplets);
  Eigen::SparseMatrix<double> Matrix_system =
      Matrix_system_half.selfadjointView<Eigen::Upper>();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Solver
  EigenCholesky choleskySolver;
  choleskySolver.compute(Matrix_system);

  // u = solution in Samplet Basis
  FMCA::Vector u = choleskySolver.solve(F_comp + beta * G_comp - N_comp);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::SparseMatrix<double> Kcomp_ss_Sparse_symm =
      Kcomp_ss_Sparse.selfadjointView<Eigen::Upper>();
  FMCA::Vector KU = Kcomp_ss_Sparse_symm * u;
  KU = hst_sources.inverseSampletTransform(KU);
  KU = hst_sources.toNaturalOrder(KU);

  //FMCA::Scalar error = (KU - analytical_sol).norm() / analytical_sol.norm();
  FMCA::Scalar error_sum = (KU - analytical_sol).sum() / P_sources.cols();
  std::cout << "Error:                                            " << error_sum
            << std::endl;
  return error_sum;
}


int main(){
    FMCA::Matrix P;
    readTXT("data/points_Halton_error100.txt", P, 2);

    int num_simulations = 100;
    FMCA::Vector error_vector(num_simulations);
    int t = P.cols()/num_simulations;
    for (int i = 0; i < num_simulations; ++i){
        int start_col = i * t;
        int end_col = start_col + t;
        // Correctly slice the matrix (assuming FMCA::Matrix supports this slicing)
        // This pseudo-code might need to be adjusted based on actual matrix library functions
        FMCA::Matrix P_sources = P.block(0, start_col, P.rows(), t);
        FMCA::Scalar error = ComputeError(P_sources);
        error_vector[i] = error;
    }
    // Write errors to a text file
    std::ofstream error_file("results_simulations/results_error_convergence.txt");
    for (double error : error_vector) {
        error_file << error << std::endl;
    }
    error_file.close();

    return 0;
}