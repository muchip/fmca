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
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/Samplets/adaptiveTreeSearch.h"
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
using EigenCholesky =
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>;
using namespace std;

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> Sparse;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
  readTXT("data/vertices_square.txt", P_sources, 2);
  readTXT("data/quadrature3_points_square.txt", P_quad, 2);
  readTXT("data/quadrature3_weights_square.txt", w_vec);
  readTXT("data/quadrature_border.txt", P_quad_border, 2);
  readTXT("data/weights_border.txt", w_vec_border);
  readTXT("data/normals.txt", Normals, 2);
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
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 3;
  const FMCA::Scalar threshold_kernel = 1e-4;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 4;
  const FMCA::Scalar sigma = 2 / (sqrt(P_sources.cols()));
  const FMCA::Scalar beta = 1000;
  const std::string kernel_type = "MATERN32";
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  FMCA::Scalar threshold_gradKernel = 10 * threshold_kernel / sigma;
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
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Kernel source-source compression
  const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
  const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources,
                                                 kernel_funtion_ss);
  std::cout << "Kernel Source-Source" << std::endl;
  Sparse Kcomp_ss_Sparse = createCompressedSparseMatrixSymmetric(
      kernel_funtion_ss, mat_eval_kernel_ss, hst_sources, eta, threshold_kernel,
      P_sources.cols());
  Sparse pattern_Kss = CreatePatternSymmetric (P_sources.cols(), hst_sources, eta, threshold_kernel);
  std::cout << pattern_Kss.size() << std::endl;
      /*
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
        Sparse Wcomp_Sparse = createCompressedWeights(
            mat_eval_weights, hst_quad, eta, threshold_weights, P_quad.cols());
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Weights border compression
        FMCA::Vector w_perm_border =
       hst_quad_border.toClusterOrder(w_vec_border);
        FMCA::SparseMatrix<FMCA::Scalar> W_border(w_perm_border.size(),
                                                  w_perm_border.size());
        for (int i = 0; i < w_perm_border.size(); ++i) {
          W_border.insert(i, i) = w_perm_border(i);
        }
        FMCA::SparseMatrixEvaluator mat_eval_weights_border(W_border);
        std::cout << "Weights border" << std::endl;
        Sparse Wcomp_border_Sparse =
            createCompressedWeights(mat_eval_weights_border, hst_quad_border,
       eta, threshold_weights, P_quad_border.cols());
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Kernel (sources,quad) compression
        const FMCA::CovarianceKernel kernel_funtion(kernel_type, sigma);
        const usMatrixEvaluatorKernel mat_eval_kernel(mom_sources, mom_quad,
       kernel_funtion);
        FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
        std::cout << "Kernel Source-Quad" << std::endl;
        Sparse Kcomp_Sparse = createCompressedSparseMatrixUnSymmetric(
            kernel_funtion, mat_eval_kernel, hst_sources, hst_quad, eta,
            threshold_kernel, P_sources.cols(), P_quad.cols());
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Penalty: construct the matrix M_b = K_{Psources, Pquad_border} *
       W_border * K_{Psources, Pquad_border}.transpose() const
       FMCA::CovarianceKernel kernel_funtion_sources_quadborder(kernel_type,
       sigma); const usMatrixEvaluatorKernel mat_eval_kernel_sources_quadborder(
            mom_sources, mom_quad_border, kernel_funtion_sources_quadborder);
        FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
            Kcomp_sources_quadborder;
        std::cout << "Kernel Source-QuadBorder" << std::endl;
        Sparse Kcomp_sources_quadborder_Sparse =
            createCompressedSparseMatrixUnSymmetric(
                kernel_funtion_sources_quadborder,
       mat_eval_kernel_sources_quadborder, hst_sources, hst_quad_border, eta,
       threshold_kernel, P_sources.cols(), P_quad_border.cols());
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
        double mult_time_eigen_penalty = T.toc();
        std::cout << "multiplication time penalty:              "
                  << mult_time_eigen_penalty << std::endl;
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // create stiffness and Neumann term
        Eigen::SparseMatrix<double> stiffness(P_sources.cols(),
       P_sources.cols()); stiffness.setZero();

        Eigen::SparseMatrix<double> grad_times_normal(P_sources.cols(),
                                                      P_quad_border.cols());
        grad_times_normal.setZero();
        FMCA::Scalar anz_gradkernel = 0;
        for (FMCA::Index i = 0; i < DIM; ++i) {
          std::cout << std::string(80, '-') << std::endl;
          const FMCA::GradKernel function(kernel_type, sigma, 1, i);
          ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
          /// stiffness
          const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
          FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
       Scomp; std::cout << "GradK" << std::endl; Sparse Scomp_Sparse =
       createCompressedSparseMatrixUnSymmetric( function, mat_eval, hst_sources,
       hst_quad, eta, threshold_gradKernel, P_sources.cols(), P_quad.cols());

          T.tic();
          Eigen::SparseMatrix<double> gradk =
              Scomp_Sparse * (Wcomp_Sparse * Scomp_Sparse.transpose()).eval();
          double mult_time_eigen = T.toc();
          std::cout << "eigen mult time component " << i << " "
                    << mult_time_eigen << std::endl;
          gradk.makeCompressed();
          stiffness += gradk.selfadjointView<Eigen::Upper>();
          ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
          /// Neumann
          const usMatrixEvaluator mat_eval_neumann(mom_sources, mom_quad_border,
                                                   function);
          FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
              Scomp_neumann;
          std::cout << "Neumann" << std::endl;
          Sparse Scomp_Sparse_neumann = createCompressedSparseMatrixUnSymmetric(
              function, mat_eval_neumann, hst_sources, hst_quad_border, eta,
              threshold_gradKernel, P_sources.cols(), P_quad_border.cols());

          Eigen::SparseMatrix<double> gradk_n =
              NeumannNormalMultiplication(Scomp_Sparse_neumann, Normals.row(i));
          gradk_n.makeCompressed();
          grad_times_normal += gradk_n;
        }
        anz_gradkernel = anz_gradkernel / DIM;
        std::cout << std::string(80, '-') << std::endl;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Final Neumann Term
        Sparse Neumann =
            grad_times_normal *
       (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
                                 Kcomp_sources_quadborder_Sparse.transpose())
                                    .eval();
        // Nitsche’s Term
        Sparse Neumann_Nitsche = Neumann.transpose();
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // reorder the boundary conditions and the rhs to follow the samplets
       order FMCA::Vector u_bc_reordered = hst_quad_border.toClusterOrder(u_bc);
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
                              (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>()
       * hst_quad_border.sampletTransform(u_bc_reordered)) .eval();
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // N_comp = right hand side Nitsche
        FMCA::Vector N_comp =
            grad_times_normal *
       (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
                                 hst_quad_border.sampletTransform(u_bc_reordered))
                                    .eval();
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        Sparse Matrix_system =
            stiffness + beta * Mb - (Neumann + Neumann_Nitsche);
       ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Solver
        EigenCholesky choleskySolver;
        choleskySolver.compute(Matrix_system);
        std::cout << "Matrix of the system is PD:              "
                  << (choleskySolver.info() == Eigen::Success) << std::endl;
        // u = solution in Samplet Basis
        FMCA::Vector u = choleskySolver.solve(F_comp + beta * G_comp - N_comp);

      */
      return 0;
}
