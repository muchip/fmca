#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
// ##############################
#include <Eigen/CholmodSupport>
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
#include "../FMCA/src/util/Plotter.h"
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
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using EigenCholesky =
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>;

int main() {
  // POINTS
  FMCA::Tictoc T;
  FMCA::Matrix P_sources_total;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_quad_border;
  FMCA::Matrix Peval;
  FMCA::Matrix Normals;
  FMCA::Vector w_vec;
  FMCA::Vector w_vec_border;

  // readTXT("data/Halton01_1M.txt", P_sources_total, DIM);
  //   P_sources = P_sources_total.leftCols(n);
  // FMCA::Scalar sigma_factor = sqrt(1./10) ;

  //   "data/subset_10000.txt", "data/subset_20000.txt", "data/subset_80000.txt"
  std::vector<std::string> points = {"data/square01_130k.txt",
                                     "data/square01_525k.txt"};
  for (std::string n : points) {
    readTXT(n, P_sources, DIM);
    readTXT("data/quadrature3_points_Unitsquare130k.txt", P_quad, DIM);
    readTXT("data/quadrature3_weights_Unitsquare130k.txt", w_vec);

    readTXT("data/quadrature01_border130k.txt", P_quad_border, DIM);
    readTXT("data/weights01_border130k.txt", w_vec_border);
    readTXT("data/normals01_130k.txt", Normals, DIM);
    readTXT("data/square01_33k.txt", Peval, DIM);
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
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Parameters
    const FMCA::Scalar eta = 1. / DIM;
    const FMCA::Index dtilde = 5;
    const FMCA::Scalar threshold_kernel = 1e-6;
    const FMCA::Scalar threshold_gradKernel = 1e-4;
    const FMCA::Scalar threshold_weights = 0;
    const FMCA::Scalar MPOLE_DEG = 2 * (dtilde - 1);
    const std::string kernel_type = "matern32";
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
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FMCA::Vector w_perm = w_vec;
    // // hst_quad.toClusterOrder(w_vec);
    // FMCA::SparseMatrix<FMCA::Scalar> W(w_perm.size(), w_perm.size());
    // for (int i = 0; i < w_perm.size(); ++i) {
    //   W.insert(i, i) = w_perm(i);
    // }
    // const FMCA::SparseMatrixEvaluator mat_eval_weights(W);
    // FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp;
    // std::cout << "Weights" << std::endl;
    // T.tic();
    // Eigen::SparseMatrix<double> W_Sparse = createCompressedWeights(
    //     mat_eval_weights, hst_quad, eta, 0, w_vec.rows());
    // std::cout << "Sum Weights:    " << w_vec.sum() << std::endl;
    // FMCA::Scalar weights_time = T.toc();
    // std::cout << "Comrpession time weights:  " << weights_time << std::endl;
    // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // std::cout << std::string(80, '-') << std::endl;
    // FMCA::SparseMatrix<FMCA::Scalar> W_border(w_vec_border.size(),
    //                                           w_vec_border.size());
    // for (int i = 0; i < w_vec_border.size(); ++i) {
    //   W_border.insert(i, i) = w_vec_border(i);
    // }
    // const FMCA::SparseMatrixEvaluator mat_eval_weights_border(W_border);
    // FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp_border;
    // std::cout << "Weights_border" << std::endl;
    // T.tic();
    // Eigen::SparseMatrix<double> W_Sparse_border = createCompressedWeights(
    //     mat_eval_weights_border, hst_quad_border, eta, 0,
    //     w_vec_border.rows());
    // std::cout << "Sum Weights Border:    " << w_vec_border.sum() <<
    // std::endl; FMCA::Scalar weights_border_time = T.toc(); std::cout <<
    // "Comrpession time weights border:  " << weights_border_time
    //           << std::endl;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    FMCA::Vector minDistance = minDistanceVector(hst_sources, P_sources);
    FMCA::Scalar sigma_mean = minDistance.maxCoeff();
    std::cout << "fill dist:                     " << sigma_mean << std::endl;

    // std::vector<FMCA::Scalar> betas = {50000, 1000000};
    FMCA::Scalar sigma = 0.05;
    FMCA::Scalar beta = 1e6;

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
    std::cout << "threshold kernel =                                "
              << threshold_kernel << std::endl;
    std::cout << "threshold grad =                                  "
              << threshold_gradKernel << std::endl;
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
    const auto& triplets = Kcomp_ss.triplets();

    // compression error
    int N_rows_ss = P_sources.cols();
    FMCA::Vector x(N_rows_ss), y1(N_rows_ss), y2(N_rows_ss);
    FMCA::Scalar err = 0;
    FMCA::Scalar nrm = 0;
    for (auto i = 0; i < 100; ++i) {
      FMCA::Index index = rand() % N_rows_ss;
      x.setZero();
      x(index) = 1;
      FMCA::Vector col = kernel_funtion_ss.eval(
          P_sources, P_sources.col(hst_sources.indices()[index]));
      y1 = col(Eigen::Map<const FMCA::iVector>(hst_sources.indices(),
                                               hst_sources.block_size()));
      x = hst_sources.sampletTransform(x);
      y2.setZero();
      for (const auto& i : triplets) {
        y2(i.row()) += i.value() * x(i.col());
        if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
      }
      y2 = hst_sources.inverseSampletTransform(y2);
      err += (y1 - y2).squaredNorm();
      nrm += y1.squaredNorm();
    }
    std::cout << "compression error =          " << sqrt(err / nrm)
              << std::endl;

    int anz = triplets.size() / P_sources.cols();
    std::cout << "Anz:                         " << anz << std::endl;

    Eigen::SparseMatrix<double> Kcomp_ss_Sparse(P_sources.cols(),
                                                P_sources.cols());
    Kcomp_ss_Sparse.setFromTriplets(triplets.begin(), triplets.end());
    Kcomp_ss_Sparse.makeCompressed();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Kernel (sources,quad) compression
    const FMCA::CovarianceKernel kernel_funtion(kernel_type, sigma);
    const usMatrixEvaluatorKernel mat_eval_kernel(mom_sources, mom_quad,
                                                  kernel_funtion);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
    std::cout << "Kernel Source-Quad" << std::endl;
    T.tic();
    Eigen::SparseMatrix<double> Kcomp_Sparse =
        createCompressedSparseMatrixUnSymmetric(
            kernel_funtion, mat_eval_kernel, hst_sources, hst_quad, eta,
            threshold_kernel, P_sources, P_quad);
    FMCA::Scalar source_quad_time = T.toc();
    std::cout << "Comrpession time:  " << source_quad_time << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Penalty: construct the matrix M_b = K_{Psources, Pquad_border} *
    // W_border
    // * K_{Psources, Pquad_border}.transpose()
    const FMCA::CovarianceKernel kernel_funtion_sources_quadborder(kernel_type,
                                                                   sigma);
    const usMatrixEvaluatorKernel mat_eval_kernel_sources_quadborder(
        mom_sources, mom_quad_border, kernel_funtion_sources_quadborder);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
        Kcomp_sources_quadborder;
    std::cout << "Kernel Source-QuadBorder" << std::endl;
    T.tic();
    Eigen::SparseMatrix<double> Kcomp_sources_quadborder_Sparse =
        createCompressedSparseMatrixUnSymmetric(
            kernel_funtion_sources_quadborder,
            mat_eval_kernel_sources_quadborder, hst_sources, hst_quad_border,
            eta, threshold_kernel, P_sources, P_quad_border);
    FMCA::Scalar source_qb_time = T.toc();
    std::cout << "Comrpession time:  " << source_qb_time << std::endl;

    T.tic();
    Sparse pattern_penalty(P_sources.cols(), P_sources.cols());
    pattern_penalty.setFromTriplets(a_priori_triplets.begin(),
                                    a_priori_triplets.end());

    formatted_sparse_multiplication_dotproduct(pattern_penalty,
                                               Kcomp_sources_quadborder_Sparse,
                                               Kcomp_sources_quadborder_Sparse);
    FMCA::Scalar mb_time = T.toc();
    std::cout << "Multiplication time:  " << mb_time << std::endl;
    Eigen::SparseMatrix<double> Mb = w_vec_border.mean() * pattern_penalty;
    // formatted_sparse_multiplication_triple_product(
    //     pattern_penalty, Kcomp_sources_quadborder_Sparse,
    //     W_Sparse_border.selfadjointView<Eigen::Upper>(),
    //     Kcomp_sources_quadborder_Sparse);
    // Eigen::SparseMatrix<double> Mb = pattern_penalty;
    // FMCA::Scalar conditionNumber_penalty = conditionNumber(beta * Mb);
    // std::cout << "conditionNumber_penalty         " <<
    // conditionNumber_penalty
    //           << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // create stiffness and Neumann term
    Eigen::SparseMatrix<double> stiffness(P_sources.cols(), P_sources.cols());
    stiffness.setZero();

    Eigen::SparseMatrix<double> grad_times_normal(P_sources.cols(),
                                                  P_quad_border.cols());
    grad_times_normal.setZero();

    for (FMCA::Index i = 0; i < DIM; ++i) {
      const FMCA::GradKernel function(kernel_type, sigma, 1, i);
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /// stiffness
      const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
      FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
      std::cout << "GradK" << std::endl;
      T.tic();
      Eigen::SparseMatrix<double> Scomp_Sparse =
          createCompressedSparseMatrixUnSymmetric(
              function, mat_eval, hst_sources, hst_quad, eta,
              threshold_gradKernel, P_sources, P_quad);
      FMCA::Scalar grad_time = T.toc();
      std::cout << "Comrpession time:  " << grad_time << std::endl;

      T.tic();
      Sparse pattern_stiffness(P_sources.cols(), P_sources.cols());
      pattern_stiffness.setFromTriplets(a_priori_triplets.begin(),
                                        a_priori_triplets.end());

      formatted_sparse_multiplication_dotproduct(pattern_stiffness,
                                                 Scomp_Sparse, Scomp_Sparse);
      FMCA::Scalar grad_mult_time = T.toc();
      std::cout << "Multiplication time:  " << grad_mult_time << std::endl;

      Eigen::SparseMatrix<double> gradk = w_vec.mean() * pattern_stiffness;
      // formatted_sparse_multiplication_triple_product(
      //     pattern_stiffness, Scomp_Sparse,
      //     W_Sparse.selfadjointView<Eigen::Upper>(), Scomp_Sparse);
      // Eigen::SparseMatrix<double> gradk = pattern_stiffness;
      stiffness += gradk;
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Neumann
      const usMatrixEvaluator mat_eval_neumann(mom_sources, mom_quad_border,
                                               function);
      FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
          Scomp_neumann;
      std::cout << "Neumann" << std::endl;
      T.tic();
      Eigen::SparseMatrix<double> Scomp_Sparse_neumann =
          createCompressedSparseMatrixUnSymmetric(
              function, mat_eval_neumann, hst_sources, hst_quad_border, eta,
              threshold_gradKernel, P_sources, P_quad_border);
      FMCA::Scalar neumann_time = T.toc();
      std::cout << "Comrpession time:  " << neumann_time << std::endl;

      Eigen::SparseMatrix<double> gradk_n =
          NeumannNormalMultiplication(Scomp_Sparse_neumann, Normals.row(i));
      gradk_n.makeCompressed();
      grad_times_normal += gradk_n;
    }
    // FMCA::Scalar conditionNumber_stiffness = conditionNumber(stiffness);
    // std::cout << "conditionNumber_stiffness         "
    //           << conditionNumber_stiffness << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Final Neumann Term
    Sparse pattern_neumann(P_sources.cols(), P_sources.cols());
    pattern_neumann.setFromTriplets(a_priori_triplets.begin(),
                                    a_priori_triplets.end());
    pattern_neumann.makeCompressed();
    formatted_sparse_multiplication_dotproduct(pattern_neumann, grad_times_normal,
                                               Kcomp_sources_quadborder_Sparse);

    Eigen::SparseMatrix<double> Neumann = w_vec_border.mean() * pattern_neumann;
    // Nitsche’s Term
    Eigen::SparseMatrix<double> Neumann_Nitsche = Neumann.transpose();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // reorder the boundary conditions and the rhs to follow the samplets
    // order
    FMCA::Vector u_bc_reordered = hst_quad_border.toClusterOrder(u_bc);
    FMCA::Vector f_reordered = hst_quad.toClusterOrder(f);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // F_comp = right hand side of the problem involving the source term f
    FMCA::Vector F_comp =
        Kcomp_Sparse * hst_quad.sampletTransform(f_reordered);
    F_comp *= w_vec.mean();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // G_comp = right hand side penalty
    FMCA::Vector G_comp = Kcomp_sources_quadborder_Sparse *
                           hst_quad_border.sampletTransform(u_bc_reordered);
    G_comp *= w_vec_border.mean();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // N_comp = right hand side Nitsche
    FMCA::Vector N_comp =
        grad_times_normal * hst_quad_border.sampletTransform(u_bc_reordered);
    N_comp *= w_vec_border.mean();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Eigen::SparseMatrix<double> Matrix_system_half =
        (stiffness + beta * Mb - (Neumann + Neumann_Nitsche));

    applyPatternAndFilter(Matrix_system_half, a_priori_triplets, 1e-8);

    Eigen::SparseMatrix<double> Matrix_system =
        Matrix_system_half.selfadjointView<Eigen::Upper>();

    std::cout << "Number of element per row system matrix:            "
              << Matrix_system.nonZeros() / Matrix_system.rows() << std::endl;
    // Check elements
    int entries_smaller_threshold = countSmallerThan(Matrix_system, 1e-8);
    std::cout << "Number of entries smaller than 1e-8:            "
              << entries_smaller_threshold << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Solver
    T.tic();
    EigenCholesky choleskySolver;
    choleskySolver.compute(Matrix_system);
    if (choleskySolver.info() != Eigen::Success) {
      throw std::runtime_error("Decomposition failed");
    }
    // u = solution in Samplet Basis
    FMCA::Vector u = choleskySolver.solve(F_comp + beta * G_comp - N_comp);
    FMCA::Scalar solver_time = T.toc();
    std::cout << "Solver time:  " << solver_time << std::endl;
    
    FMCA::Vector u_grid = hst_sources.inverseSampletTransform(u);
    u_grid = hst_sources.toNaturalOrder(u_grid);

    // std::cout
    //     << "residual error:                                  "
    //     << ((Matrix_system)*u - (F_comp + beta * G_comp - N_comp)).norm()
    //     << std::endl;
    // T.tic();
    // Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> cholmod;
    // cholmod.compute(Matrix_system);
    // FMCA::Vector u = cholmod.solve(F_comp + beta * G_comp - N_comp);
    // FMCA::Scalar solver_time = T.toc();
    // std::cout<< "Comrpession time:  " << solver_time << std::endl;

    // std::cout
    //     << "residual error cholmod: "
    //     << ((Matrix_system)*u - (F_comp + beta * G_comp - N_comp)).norm()
    //     << std::endl;

    // FMCA::Vector u_grid = hst_sources.inverseSampletTransform(u);
    // u_grid = hst_sources.toNaturalOrder(u_grid);

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
      FMCA::Plotter2D plotter;
      plotter.plotFunction("solution_exponential_Poisson.vtk", Peval, rec);

      // Analytical sol of the problem
      FMCA::Vector analytical_sol(Peval.cols());
      for (int i = 0; i < Peval.cols(); ++i) {
        double x = Peval(0, i);
        double y = Peval(1, i);
        analytical_sol[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
      }

      FMCA::Scalar error =
          (rec - analytical_sol).norm() / analytical_sol.norm();
      std::cout << "Error:                                            " << error
                << std::endl;
      std::cout << "--------------------------------------------------"
                << std::endl;
    }
  }
  return 0;
}
