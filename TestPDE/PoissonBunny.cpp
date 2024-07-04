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
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "../FMCA/src/util/Plotter3D.h"
#include "FunctionsPDE.h"
#include "read_files_txt.h"

#define DIM 3

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

int main() {
  // POINTS
  FMCA::Tictoc T;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_quad_border;
  FMCA::Matrix Peval;
  FMCA::Matrix Normals;
  FMCA::Vector w_vec;
  FMCA::Vector w_vec_border;

  readTXT("data/vertices_bunny100k.txt", P_sources, 3);
  readTXT("data/barycenters_bunny.txt", P_quad, 3);
  readTXT("data/volumes_bunny.txt", w_vec);
  readTXT("data/quadrature3_points_bunny_surface.txt", P_quad_border, 3);
  readTXT("data/quadrature3_weights_bunny_surface.txt", w_vec_border);
  readTXT("data/Int_and_Bnd_bunny.txt", Peval, 3);
  // readTXT("data/normals5_40k.txt", Normals, 3);
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
    // f[i] = 2 * FMCA_PI * FMCA_PI * sin(FMCA_PI * x) * sin(FMCA_PI * y);
    f[i] = -100;
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
  const FMCA::Scalar threshold_kernel = 1e-3;
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
  std::vector<FMCA::Scalar> sigmas = {2 * sigma_h};
  // 0.008 error is   0.000856655

  for (FMCA::Scalar sigma : sigmas) {
    // FMCA::Scalar threshold_gradKernel = threshold_kernel / (sigma * sigma);
    FMCA::Scalar threshold_gradKernel = 1e-1;
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
    Sparse pattern_penalty(P_sources.cols(), P_sources.cols());
    pattern_penalty.setFromTriplets(a_priori_triplets.begin(),
                                    a_priori_triplets.end());
    T.tic();
    pattern_penalty.makeCompressed();

    formatted_sparse_multiplication_triple_product(
        pattern_penalty, Kcomp_sources_quadborder_Sparse,
        Wcomp_border_Sparse.selfadjointView<Eigen::Upper>(),
        Kcomp_sources_quadborder_Sparse);
    // Eigen::SparseMatrix<double> Mb1 =
    //     Wcomp_border_Sparse * Kcomp_sources_quadborder_Sparse.transpose();
    // Eigen::SparseMatrix<double> Mb = Kcomp_sources_quadborder_Sparse * Mb1;
    // Mb = Mb.selfadjointView<Eigen::Upper>();
    std::cout << "Penalty created" << std::endl;
    pattern_penalty.makeCompressed();
    Eigen::SparseMatrix<double> Mb =
        pattern_penalty.selfadjointView<Eigen::Upper>();
    double mult_time_eigen_penalty = T.toc();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // create stiffness and Neumann term
    Eigen::SparseMatrix<double> stiffness(P_sources.cols(), P_sources.cols());
    stiffness.setZero();

    // Eigen::SparseMatrix<double> grad_times_normal(P_sources.cols(),
    //                                               P_quad_border.cols());
    // grad_times_normal.setZero();
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
      //  Eigen::SparseMatrix<double> gradk1 = Wcomp_Sparse *
      //          Scomp_Sparse.transpose();
      // Eigen::SparseMatrix<double> gradk = Scomp_Sparse * gradk1 ;
      formatted_sparse_multiplication_triple_product(
          pattern, Scomp_Sparse, Wcomp_Sparse.selfadjointView<Eigen::Upper>(),
          Scomp_Sparse);
      double mult_time_eigen = T.toc();
      pattern.makeCompressed();
      stiffness += pattern.selfadjointView<Eigen::Upper>();
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Neumann
    //   const usMatrixEvaluator mat_eval_neumann(mom_sources, mom_quad_border,
    //                                            function);
    //   FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
    //       Scomp_neumann;
    //   std::cout << "Neumann" << std::endl;
    //   Eigen::SparseMatrix<double> Scomp_Sparse_neumann =
    //       createCompressedSparseMatrixUnSymmetric(
    //           function, mat_eval_neumann, hst_sources, hst_quad_border, eta,
    //           threshold_gradKernel, P_sources.cols(), P_quad_border.cols());

      //   Eigen::SparseMatrix<double> gradk_n =
      //       NeumannNormalMultiplication(Scomp_Sparse_neumann,
      //       Normals.row(i));
      //   gradk_n.makeCompressed();
      //   grad_times_normal += gradk_n;
    }
    anz_gradkernel = anz_gradkernel / DIM;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // // Final Neumann Term
    // Eigen::SparseMatrix<double> Neumann =
    //     grad_times_normal *
    //     (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
    //      Kcomp_sources_quadborder_Sparse.transpose())
    //         .eval();
    // // Nitsche’s Term
    // Eigen::SparseMatrix<double> Neumann_Nitsche = Neumann.transpose();
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
    // FMCA::Vector N_comp = grad_times_normal *
    //                       (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>()
    //                       *
    //                        hst_quad_border.sampletTransform(u_bc_reordered))
    //                           .eval();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Eigen::SparseMatrix<double> Matrix_system_half =
        (stiffness + beta * Mb);  // - (Neumann + Neumann_Nitsche)
    
    applyPatternAndFilter(Matrix_system_half, a_priori_triplets, 1e-4); // remove the triplets with value less than the threshold
    
    applyPattern(Matrix_system_half, a_priori_triplets);
    Eigen::SparseMatrix<double> Matrix_system =
        Matrix_system_half.selfadjointView<Eigen::Upper>();
    // Eigen::SparseMatrix<double> Matrix_system =
    //     stiffness - (Neumann + Neumann_Nitsche) + beta * Mb;

    std::cout << "Number of element per row system matrix:            "
              << Matrix_system.nonZeros() / Matrix_system.rows() << std::endl;
    // Check elements
    int entries_smaller_threshold = countSmallerThan(Matrix_system, 1e-4);
    std::cout << "Number of entries smaller than 1e-4:            "
              << entries_smaller_threshold << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Solver
    EigenCholesky choleskySolver;
    choleskySolver.compute(Matrix_system);

    // u = solution in Samplet Basis
    FMCA::Vector u = choleskySolver.solve(F_comp + beta * G_comp);  // - N_comp
    FMCA::Vector u_grid = hst_sources.inverseSampletTransform(u);
    u_grid = hst_sources.toNaturalOrder(u_grid);

    // Check the solver
    std::cout
        << "residual error:                                  "
        << ((Matrix_system)*u - (F_comp + beta * G_comp)).norm()  // - N_comp
        << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Eigen::SparseMatrix<double> Kcomp_ss_Sparse_symm =
        Kcomp_ss_Sparse.selfadjointView<Eigen::Upper>();
    FMCA::Vector KU = Kcomp_ss_Sparse_symm * u;
    KU = hst_sources.inverseSampletTransform(KU);
    KU = hst_sources.toNaturalOrder(KU);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // plot solution
    {
      const Moments cmom(P_sources, MPOLE_DEG);
      const Moments rmom(Peval, MPOLE_DEG);
      const H2ClusterTree hct(cmom, 0, P_sources);
      const H2ClusterTree hct_eval(rmom, 0, Peval);
      const FMCA::Vector sx = hct.toClusterOrder(u_grid);
      const usMatrixEvaluatorKernel mat_eval(rmom, cmom, kernel_funtion_ss);
      FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster> hmat;
      hmat.computePattern(hct_eval, hct, eta);
      hmat.statistics();
      FMCA::Vector srec = hmat.action(mat_eval, sx);
      FMCA::Vector rec = hct_eval.toNaturalOrder(srec);
      FMCA::Plotter3D plotter;
      plotter.plotFunction("bunny.vtk", Peval, rec);
    }

  }
  return 0;
}
