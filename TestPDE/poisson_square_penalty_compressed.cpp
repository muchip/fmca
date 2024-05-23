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

// typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> Sparse;
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
  //   readTXT("data/vertices_square130k.txt", P_sources, 2);
  //   readTXT("data/quadrature3_points_square130k.txt", P_quad, 2);
  //   readTXT("data/quadrature3_weights_square130k.txt", w_vec);
  //   readTXT("data/quadrature_border130k.txt", P_quad_border, 2);
  //   readTXT("data/weights_border130k.txt", w_vec_border);
  //   readTXT("data/normals130k.txt", Normals, 2);
  readTXT("data/vertices_square_Halton5000.txt", P_sources, 2);
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

  FMCA::Vector minDistance = minDistanceVector(hst_sources, P_sources);
  auto maxElementIterator =
      std::max_element(minDistance.begin(), minDistance.end());
  FMCA::Scalar sigma_h = *maxElementIterator;
  std::cout << "fill distance:                      " << sigma_h << std::endl;
  std::vector<FMCA::Scalar> sigmas = {sigma_h, 2*sigma_h, 1.5*sigma_h};
  // 0.008 error is   0.000856655

  for (FMCA::Scalar sigma : sigmas) {
    // FMCA::Scalar threshold_gradKernel = threshold_kernel / (sigma * sigma);
    FMCA::Scalar threshold_gradKernel = 1e-2;
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
    std::cout << "MPOLE_DEG =                                       "
              << MPOLE_DEG << std::endl;
    std::cout << std::string(80, '-') << std::endl;
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
    std::cout << "Kernel Source-QuadBorder" << std::endl;
    Eigen::SparseMatrix<double> Kcomp_sources_quadborder_Sparse =
        createCompressedSparseMatrixUnSymmetric(
            kernel_funtion_sources_quadborder,
            mat_eval_kernel_sources_quadborder, hst_sources, hst_quad_border,
            eta, threshold_kernel, P_sources.cols(), P_quad_border.cols());
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
    std::cout << "multiplication time penalty:              "
              << mult_time_eigen_penalty << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // create stiffness and Neumann term
    Eigen::SparseMatrix<double> stiffness(P_sources.cols(), P_sources.cols());
    stiffness.setZero();

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
      //   Eigen::SparseMatrix<double> gradk =
      //       Scomp_Sparse * Scomp_Sparse.transpose();
      pattern *= w_vec(0);
      // applyPattern(gradk, triplets_pattern);
      double mult_time_eigen = T.toc();
      std::cout << "eigen mult time component " << i
                << "                        " << mult_time_eigen << std::endl;
      pattern.makeCompressed();
      stiffness += pattern.selfadjointView<Eigen::Upper>();
      // stiffness += gradk;
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
    std::cout << std::string(80, '-') << std::endl;

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

    // Trace
    double trace = computeTrace(Matrix_system);
    std::cout << "Trace of the system matrix:                         " << trace
              << std::endl;
    std::cout << "Number of element per row system matrix:            "
              << Matrix_system.nonZeros() / Matrix_system.rows() << std::endl;
    // Check elements
    int entries_smaller_threshold = countSmallerThan(Matrix_system, 1e-6);
    std::cout << "Number of entries smaller than threshold:            "
              << entries_smaller_threshold << std::endl;
    // check Symmetry of Matrix_system
    bool issymmetric = isSymmetric(Matrix_system);
    std::cout << "Matrix of the system is symmetric:       " << issymmetric
              << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    FMCA::IO::print2spascii("galerkin_square", Matrix_system, "w");
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Solver
    EigenCholesky choleskySolver;
    choleskySolver.compute(Matrix_system);
    std::cout << "Matrix of the system is PD:              "
              << (choleskySolver.info() == Eigen::Success) << std::endl;

    // u = solution in Samplet Basis
    FMCA::Vector u = choleskySolver.solve(F_comp + beta * G_comp - N_comp);
    // FMCA::Vector u_natural_order = hst_sources.toNaturalOrder(u);
    // FMCA::IO::print2m(
    //     "results_matlab/solution_square_penalty_compressed_SampletBasis.m",
    //     "sol_square_penalty_compressed_SampletBasis", u, "w");

    // Check the solver
    std::cout << "residual error:                                  "
              << ((Matrix_system)*u - (F_comp + beta * G_comp - N_comp)).norm()
              << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Eigen::SparseMatrix<double> Kcomp_ss_Sparse_symm =
        Kcomp_ss_Sparse.selfadjointView<Eigen::Upper>();
    FMCA::Vector KU = Kcomp_ss_Sparse_symm * u;
    KU = hst_sources.inverseSampletTransform(KU);
    KU = hst_sources.toNaturalOrder(KU);
    // KU_transformed(inversePermutationVector(hst_sources));
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    FMCA::Scalar error = (KU - analytical_sol).norm() / analytical_sol.norm();
    std::cout << "Error:                                            " << error
              << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // FMCA::Vector absolute_error(P_sources.cols());
    // for (int i = 0; i < P_sources.cols(); ++i) {
    //   absolute_error[i] = abs(KU[i] - analytical_sol[i]);
    // }
    // FMCA::IO::print2m("results_matlab/solution_square_penalty_compressed.m",
    //                   "sol_square_penalty_compressed", KU, "w");
    // FMCA::IO::print2m(
    //     "results_matlab/error_square_penalty_absolute_compressed.m",
    //     "absolute_error_penalty_compressed", absolute_error, "w");
    // {
    //   FMCA::Matrix P_sources_3d(3, P_sources.cols());
    //   for (int i = 0; i < P_sources.cols(); ++i) {
    //     P_sources_3d(0, i) = P_sources(0, i);
    //     P_sources_3d(1, i) = P_sources(1, i);
    //     P_sources_3d(2, i) = 0;
    //   }
    //   Eigen::Vector3d min = {-1, -1, 0};
    //   Eigen::Vector3d max = {1, 1, 0};
    //   FMCA::Grid3D grid(min, max, 100, 100, 100);
    //   const Moments cmom(P_sources_3d, MPOLE_DEG);
    //   const FMCA::Matrix Peval = grid.P();
    //   const Moments rmom(Peval, MPOLE_DEG);
    //   const H2ClusterTree hct(cmom, 0, P_sources_3d);
    //   const H2ClusterTree hct_eval(rmom, 0, Peval);
    //   const FMCA::Vector sx = hct.toClusterOrder(KU);
    //   const usMatrixEvaluatorKernel mat_eval(rmom, cmom, kernel_funtion_ss);
    //   FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster> hmat;
    //   hmat.computePattern(hct_eval, hct, eta);
    //   hmat.statistics();
    //   FMCA::Vector srec = hmat.action(mat_eval, sx);
    //   FMCA::Vector rec = hct_eval.toNaturalOrder(srec);
    //   grid.plotFunction("square_Halton.vtk", rec);
    // }

    // std::vector<const H2SampletTree *> adaptive_tree =
    //     adaptiveTreeSearch(hst_sources, u, 1e-4 * u.squaredNorm());
    // const FMCA::Index nclusters =
    //     std::distance(hst_sources.begin(), hst_sources.end());

    // FMCA::Vector thres_tdata = u;
    // thres_tdata.setZero();
    // FMCA::Index nnz = 0;
    // for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    //   if (adaptive_tree[i] != nullptr) {
    //     const H2SampletTree &node = *(adaptive_tree[i]);
    //     const FMCA::Index ndist =
    //         node.is_root() ? node.Q().cols() : node.nsamplets();
    //     thres_tdata.segment(node.start_index(), ndist) =
    //         u.segment(node.start_index(), ndist);
    //     nnz += ndist;
    //   }
    // }
    // std::cout << "active coefficients: " << nnz << " / " << P_sources.cols()
    //           << std::endl;
    // std::cout << "tree error: " << (thres_tdata - u).norm() / u.norm()
    //           << std::endl;

    // std::vector<FMCA::Matrix> bbvec_active;
    // for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    //   if (adaptive_tree[i] != nullptr) {
    //     const H2SampletTree &node = *(adaptive_tree[i]);
    //     bbvec_active.push_back(node.bb());
    //   }
    // }

    // FMCA::IO::plotBoxes2D("active_boxes_Poisson_square.vtk", bbvec_active);
  }
  return 0;
}
