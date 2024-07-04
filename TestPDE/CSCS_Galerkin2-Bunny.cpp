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
#include "../FMCA/src/util/Plotter3D.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
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

  std::string outputNamePlot = "weakBunny.vtk";

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
    f[i] = -100;
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
    std::cout << "sigma =                                           " << sigma
              << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Kernel (source-source) compression --> KCompressed_source
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

    ////////////////////////////////// compression error
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
      for (const auto &i : triplets) {
        y2(i.row()) += i.value() * x(i.col());
        if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
      }
      y2 = hst_sources.inverseSampletTransform(y2);
      err += (y1 - y2).squaredNorm();
      nrm += y1.squaredNorm();
    }
    std::cout << "compression error =          " << sqrt(err / nrm)
              << std::endl;
    ;
    //////////////////////////////////////////////////

    int anz = triplets.size() / P_sources.cols();
    std::cout << "Anz:                            " << anz << std::endl;

    Eigen::SparseMatrix<double> KCompressed_source(P_sources.cols(),
                                                   P_sources.cols());
    KCompressed_source.setFromTriplets(triplets.begin(), triplets.end());
    KCompressed_source.makeCompressed();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Weights compression --> WCompressed
    FMCA::Vector w_perm = hst_quad.toClusterOrder(w_vec);
    FMCA::SparseMatrix<FMCA::Scalar> W(w_perm.size(), w_perm.size());
    for (int i = 0; i < w_perm.size(); ++i) {
      W.insert(i, i) = w_perm(i);
    }
    const FMCA::SparseMatrixEvaluator mat_eval_weights(W);
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp;
    std::cout << "Weights" << std::endl;

    Eigen::SparseMatrix<double> WCompressed = createCompressedWeights(
        mat_eval_weights, hst_quad, eta, threshold_weights, P_quad.cols());
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Weights border compression --> WCompressed_border
    FMCA::Vector w_perm_border = hst_quad_border.toClusterOrder(w_vec_border);
    FMCA::SparseMatrix<FMCA::Scalar> W_border(w_perm_border.size(),
                                              w_perm_border.size());
    for (int i = 0; i < w_perm_border.size(); ++i) {
      W_border.insert(i, i) = w_perm_border(i);
    }
    FMCA::SparseMatrixEvaluator mat_eval_weights_border(W_border);
    std::cout << "Weights border" << std::endl;

    Eigen::SparseMatrix<double> WCompressed_border =
        createCompressedWeights(mat_eval_weights_border, hst_quad_border, eta,
                                threshold_weights, P_quad_border.cols());
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Kernel (sources,quad) compression
    const FMCA::CovarianceKernel kernel_funtion(kernel_type, sigma);
    const usMatrixEvaluatorKernel mat_eval_kernel(mom_sources, mom_quad,
                                                  kernel_funtion);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
    std::cout << "Kernel Source-Quad" << std::endl;

    Eigen::SparseMatrix<double> KCompressed_sourcequad =
        createCompressedSparseMatrixUnSymmetric(
            kernel_funtion, mat_eval_kernel, hst_sources, hst_quad, eta,
            threshold_kernel, P_sources, P_quad);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Penalty: K (sources, quad_border) * W_border * K
    // (sources, quad_border).transpose()
    const FMCA::CovarianceKernel kernel_funtion_sources_quadborder(kernel_type,
                                                                   sigma);
    const usMatrixEvaluatorKernel mat_eval_kernel_sources_quadborder(
        mom_sources, mom_quad_border, kernel_funtion_sources_quadborder);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
        Kcomp_sources_quadborder;
    std::cout << "Kernel Source-QuadBorder" << std::endl;

    Eigen::SparseMatrix<double> KCompressed_source_quadborder =
        createCompressedSparseMatrixUnSymmetric(
            kernel_funtion_sources_quadborder,
            mat_eval_kernel_sources_quadborder, hst_sources, hst_quad_border,
            eta, threshold_kernel, P_sources, P_quad_border);
    T.tic();
    Sparse pattern_penalty(P_sources.cols(), P_sources.cols());
    pattern_penalty.setFromTriplets(a_priori_triplets.begin(),
                                    a_priori_triplets.end());
    pattern_penalty.makeCompressed();
    formatted_sparse_multiplication_triple_product(
        pattern_penalty, KCompressed_source_quadborder, WCompressed_border,
        KCompressed_source_quadborder);
    //   Eigen::SparseMatrix<double> mult_penalty =
    //       KCompressed_source_quadborder *
    //       (WCompressed_border *
    //       KCompressed_source_quadborder.transpose()).eval();
    Eigen::SparseMatrix<double> Penalty =
        pattern_penalty.selfadjointView<Eigen::Upper>();
    double mult_time_eigen_penalty = T.toc();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // create Stiffness and Neumann term
    Eigen::SparseMatrix<double> Stiffness(P_sources.cols(), P_sources.cols());
    Stiffness.setZero();

    // Eigen::SparseMatrix<double> GradNormal(P_sources.cols(),
    //                                        P_quad_border.cols());
    // GradNormal.setZero();

    FMCA::Scalar anz_gradkernel = 0;
    for (FMCA::Index i = 0; i < DIM; ++i) {
      const FMCA::GradKernel function(kernel_type, sigma, 1, i);
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /// Stiffness
      const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
      FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
      std::cout << "GradK" << std::endl;

      Eigen::SparseMatrix<double> GradKCompressed =
          createCompressedSparseMatrixUnSymmetric(
              function, mat_eval, hst_sources, hst_quad, eta,
              threshold_gradKernel, P_sources, P_quad);

      T.tic();
      Sparse pattern(P_sources.cols(), P_sources.cols());
      pattern.setFromTriplets(a_priori_triplets.begin(),
                              a_priori_triplets.end());
      pattern.makeCompressed();
      // Eigen::SparseMatrix<double> mult_gradk =
      //     GradKCompressed * (WCompressed *
      //     GradKCompressed.transpose()).eval();
      formatted_sparse_multiplication_triple_product(
          pattern, GradKCompressed, WCompressed, GradKCompressed);
      // pattern *= w_vec(0);
      double mult_time_eigen = T.toc();
      Stiffness += pattern.selfadjointView<Eigen::Upper>();
      // .selfadjointView<Eigen::Upper>()
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }
    anz_gradkernel = anz_gradkernel / DIM;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // reorder the boundary conditions and the rhs to follow the samplets order
    FMCA::Vector u_bc_reordered = hst_quad_border.toClusterOrder(u_bc);
    FMCA::Vector f_reordered = hst_quad.toClusterOrder(f);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FCompressed = right hand side
    FMCA::Vector FCompressed =
        KCompressed_sourcequad * (WCompressed.selfadjointView<Eigen::Upper>() *
                                  hst_quad.sampletTransform(f_reordered))
                                     .eval();
    std::cout << "F compressed done" << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // GCompressed = right hand side penalty
    FMCA::Vector GCompressed =
        KCompressed_source_quadborder *
        (WCompressed_border.selfadjointView<Eigen::Upper>() *
         hst_quad_border.sampletTransform(u_bc_reordered))
            .eval();
    std::cout << "G compressed done" << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Eigen::SparseMatrix<double> Matrix_system_half =
        (Stiffness + beta * Penalty);
    std::cout << "Matrix_system_half done" << std::endl;
    applyPattern(Matrix_system_half, a_priori_triplets);
    std::cout << "applyPattern done" << std::endl;
    Eigen::SparseMatrix<double> Matrix_system =
        Matrix_system_half.selfadjointView<Eigen::Upper>();

    FMCA::Vector Matrix_system_diagonal = Matrix_system.diagonal();
    std::cout << "Min element diagonal:                               "
              << Matrix_system_diagonal.minCoeff() << std::endl;

    std::cout << "Number of element per row system matrix:            "
              << Matrix_system.nonZeros() / Matrix_system.rows() << std::endl;
    // Check elements
    int entries_smaller_threshold = countSmallerThan(Matrix_system, 1e-6);
    std::cout << "Number of entries smaller than threshold:            "
              << entries_smaller_threshold << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    EigenCholesky choleskySolver;
    choleskySolver.compute(Matrix_system);
    T.tic();
    FMCA::Vector u =
        choleskySolver.solve(FCompressed + beta * GCompressed);
    FMCA::Scalar solver_time_chol = T.toc();
    std::cout
        << "solver time SimplicialLDLT:                                    "
        << solver_time_chol << std::endl;

    std::cout
        << "residual error SimplicialLDLT:                                 "
        << ((Matrix_system)*u -
            (FCompressed + beta * GCompressed))
               .norm()
        << std::endl;

  FMCA::Vector u_grid = hst_sources.inverseSampletTransform(u);
  u_grid = hst_sources.toNaturalOrder(u_grid);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   // Cumpute the solution K*u with the compression of the kernel matrix K
//   Eigen::SparseMatrix<double> KCompressed_source_symm =
//       KCompressed_source.selfadjointView<Eigen::Upper>();
//   FMCA::Vector KU = KCompressed_source_symm * u;
//   KU = hst_sources.inverseSampletTransform(KU);
//   // Numerical solution
//   KU = hst_sources.toNaturalOrder(KU);
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
    plotter.plotFunction(outputNamePlot, Peval, rec);

  }
  }
  return 0;
}
