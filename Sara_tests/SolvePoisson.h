#include <cstdlib>
#include <iostream>
// ##############################
#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "../FMCA/GradKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/Samplets/adaptiveTreeSearch.h"
#include "../FMCA/src/util/Grid2D.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "FunctionsPDE.h"
#include "NeumanNormalMultiplication.h"

FMCA::Vector SolvePoisson(
    const FMCA::Scalar &DIM, FMCA::Matrix &P_sources, FMCA::Matrix &P_quad,
    FMCA::Vector &w_vec, FMCA::Matrix &P_quad_border,
    FMCA::Vector &w_vec_border, FMCA::Matrix &Normals, FMCA::Vector &u_bc,
    FMCA::Vector &f, const FMCA::Scalar &sigma, const FMCA::Scalar &eta, const FMCA::Index &dtilde,
    const FMCA::Scalar &threshold_kernel,
    const FMCA::Scalar &threshold_gradKernel,
    const FMCA::Scalar &threshold_weights, const FMCA::Scalar &MPOLE_DEG,
    const FMCA::Scalar &beta, const std::string &kernel_type) {
  FMCA::Tictoc T;
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

    // FMCA::Scalar threshold_gradKernel = threshold_kernel / (sigma * sigma);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Number of source points =                         "
              << P_sources.cols() << std::endl;
    std::cout << "Number of quad points =                           "
              << P_quad.cols() << std::endl;
    std::cout << "Number of quad points border =                    "
              << P_quad_border.cols() << std::endl;
    std::cout << "sigma =                                           " << sigma
              << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Kernel source-source compression
    const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
    const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources,
                                                   kernel_funtion_ss);
    std::cout << "Kernel Source-Source" << std::endl;
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
    Kcomp_ss.init(hst_sources, eta, threshold_kernel);
    std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets =
        Kcomp_ss.a_priori_pattern_triplets();
    Kcomp_ss.compress(mat_eval_kernel_ss);
    const auto &triplets = Kcomp_ss.triplets();
    int anz = triplets.size() / P_sources.cols();
    std::cout << "Anz:                         " << anz << std::endl;

    Eigen::SparseMatrix<double> Kcomp_ss_Sparse(P_sources.cols(),
                                                P_sources.cols());
    Kcomp_ss_Sparse.setFromTriplets(triplets.begin(), triplets.end());
    Kcomp_ss_Sparse.makeCompressed();
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
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Weights border compression
    FMCA::Vector w_perm_border = hst_quad_border.toClusterOrder(w_vec_border);
    FMCA::SparseMatrix<FMCA::Scalar> W_border(w_perm_border.size(),
                                              w_perm_border.size());
    for (int i = 0; i < w_perm_border.size(); ++i) {
      W_border.insert(i, i) = w_perm_border(i);
    }
    FMCA::SparseMatrixEvaluator mat_eval_weights_border(W_border);
    std::cout << "Weights border" << std::endl;
    Eigen::SparseMatrix<double> Wcomp_border_Sparse =
        createCompressedWeights(mat_eval_weights_border, hst_quad_border, eta,
                                threshold_weights, P_quad_border.cols());
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Kernel (sources,quad) compression
    const FMCA::CovarianceKernel kernel_funtion(kernel_type, sigma);
    const usMatrixEvaluatorKernel mat_eval_kernel(mom_sources, mom_quad,
                                                  kernel_funtion);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
    std::cout << "Kernel Source-Quad" << std::endl;
    Eigen::SparseMatrix<double> Kcomp_Sparse =
        createCompressedSparseMatrixUnSymmetric(
            kernel_funtion, mat_eval_kernel, hst_sources, hst_quad, eta,
            threshold_gradKernel, P_sources.cols(), P_quad.cols());
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Penalty: construct the matrix M_b = K_{Psources, Pquad_border} * W_border
    // * K_{Psources, Pquad_border}.transpose()
    const FMCA::CovarianceKernel kernel_funtion_sources_quadborder(kernel_type,
                                                                   sigma);
    const usMatrixEvaluatorKernel mat_eval_kernel_sources_quadborder(
        mom_sources, mom_quad_border, kernel_funtion_sources_quadborder);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
        Kcomp_sources_quadborder;
    std::cout << "Kernel Source-QuadBorder" << std::endl;
    Eigen::SparseMatrix<double> Kcomp_sources_quadborder_Sparse =
        createCompressedSparseMatrixUnSymmetric(
            kernel_funtion_sources_quadborder,
            mat_eval_kernel_sources_quadborder, hst_sources, hst_quad_border,
            eta, threshold_kernel, P_sources.cols(), P_quad_border.cols());
    T.tic();
    Eigen::SparseMatrix<double> Mb =
        Kcomp_sources_quadborder_Sparse *
        (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
         Kcomp_sources_quadborder_Sparse.transpose())
            .eval();
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
      std::cout << "GradK" << std::endl;
      Eigen::SparseMatrix<double> Scomp_Sparse =
          createCompressedSparseMatrixUnSymmetric(
              function, mat_eval, hst_sources, hst_quad, eta,
              threshold_gradKernel, P_sources.cols(), P_quad.cols());

      T.tic();
      Sparse pattern(P_sources.cols(), P_sources.cols());
      pattern.setFromTriplets(a_priori_triplets.begin(),
                              a_priori_triplets.end());
      pattern.makeCompressed();
      //   Eigen::SparseMatrix<double> gradk =
      //       Scomp_Sparse * (Wcomp_Sparse * Scomp_Sparse.transpose()).eval();
      formatted_sparse_multiplication_dotproduct(pattern, Scomp_Sparse,
                                                 Scomp_Sparse);
      pattern *= w_vec(0);
      double mult_time_eigen = T.toc();
      pattern.makeCompressed();
      stiffness += pattern.selfadjointView<Eigen::Upper>();
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Neumann
      const usMatrixEvaluator mat_eval_neumann(mom_sources, mom_quad_border,
                                               function);
      FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
          Scomp_neumann;
      std::cout << "Neumann" << std::endl;
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
        grad_times_normal *
        (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
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
    FMCA::Vector N_comp = grad_times_normal *
                          (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
                           hst_quad_border.sampletTransform(u_bc_reordered))
                              .eval();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Eigen::SparseMatrix<double> Matrix_system_half =
        (stiffness + beta * Mb - (Neumann + Neumann_Nitsche));
    applyPattern(Matrix_system_half, a_priori_triplets);
    Eigen::SparseMatrix<double> Matrix_system =
        Matrix_system_half.selfadjointView<Eigen::Upper>();
    // Eigen::SparseMatrix<double> Matrix_system =
    //     stiffness - (Neumann + Neumann_Nitsche) + beta * Mb;

    // Trace
    // double trace = computeTrace(Matrix_system);
    // std::cout << "Trace of the system matrix:                         " <<
    // trace
    //           << std::endl;
    std::cout << "Number of element per row system matrix:            "
              << Matrix_system.nonZeros() / Matrix_system.rows() << std::endl;
    // Check elements
    int entries_smaller_threshold = countSmallerThan(Matrix_system, 1e-8);
    std::cout << "Number of entries smaller than threshold:            "
              << entries_smaller_threshold << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Solver
    EigenCholesky choleskySolver;
    choleskySolver.compute(Matrix_system);
    // u = solution in Samplet Basis
    FMCA::Vector u = choleskySolver.solve(F_comp + beta * G_comp - N_comp);
    std::cout << "residual error:                                  "
              << ((Matrix_system)*u - (F_comp + beta * G_comp - N_comp)).norm()
              << std::endl;
    return u;
}