/* This header is the solver for the Poisson eqiuation using Samplet Matrix
Compression. It returns the solution u in samplet basis (not in the natural
order!). See PoissonSquareCompressed.cpp for an application */

#include <algorithm>
#include <cstdlib>
#include <iostream>
// ##############################

#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

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

/* The inputs are described:
- DIM: dimension of the domain Omega, 2 for 2D and 3 fro 3D
- P_sources: set of points where the kernel are generated (DIM, N)
- P_quad: set of quadrature points for the entire domain Omega (DIM, N_QUAD)
- w_vec: vector of weights associated to the quadrature points P_quad
- P_quad_border: set of quadrature points for the border \partial Omega (DIM,
N_QUAD_BORDER)
- w_vec_border: vector of weights associated to the quadrature points
P_quad_border
- Normals: external normal vector at each P_quad_border point (DIM,
N_QUAD_BORDER)
- u_bc: vector of boundary conditions at the P_quad_border
- f: vector of source of the pde at P_quad
- The others are Samplets Parameters.
*/

FMCA::Vector SolvePoisson(
    const FMCA::Scalar &DIM, FMCA::Matrix &P_sources, FMCA::Matrix &P_quad,
    FMCA::Vector &w_vec, FMCA::Matrix &P_quad_border,
    FMCA::Vector &w_vec_border, FMCA::Matrix &Normals, FMCA::Vector &u_bc,
    FMCA::Vector &f, const FMCA::Scalar &sigma, const FMCA::Scalar &eta,
    const FMCA::Index &dtilde, const FMCA::Scalar &threshold_kernel,
    const FMCA::Scalar &threshold_gradKernel,
    const FMCA::Scalar &threshold_weights, const FMCA::Scalar &MPOLE_DEG,
    const FMCA::Scalar &beta, const std::string &kernel_type) {
  FMCA::Tictoc T;
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
  // a priori triplets
  std::vector<Eigen::Triplet<FMCA::Scalar>> a_priori_triplets =
      Kcomp_ss.a_priori_pattern_triplets();
  Kcomp_ss.compress(mat_eval_kernel_ss);
  const auto &triplets = Kcomp_ss.triplets();
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
    for (const auto &i : triplets) {
      y2(i.row()) += i.value() * x(i.col());
      if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
    }
    y2 = hst_sources.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  std::cout << "compression error =          " << sqrt(err / nrm) << std::endl;
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
  // Penalty: K (sources, quad_border) * W_border * K(sources,
  // quad_border).transpose()
  const FMCA::CovarianceKernel kernel_funtion_sources_quadborder(kernel_type,
                                                                 sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel_sources_quadborder(
      mom_sources, mom_quad_border, kernel_funtion_sources_quadborder);
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
      Kcomp_sources_quadborder;
  std::cout << "Kernel Source-QuadBorder" << std::endl;

  Eigen::SparseMatrix<double> KCompressed_source_quadborder =
      createCompressedSparseMatrixUnSymmetric(
          kernel_funtion_sources_quadborder, mat_eval_kernel_sources_quadborder,
          hst_sources, hst_quad_border, eta, threshold_kernel, P_sources,
          P_quad_border);
  T.tic();
  Sparse pattern_penalty(P_sources.cols(), P_sources.cols());
  pattern_penalty.setFromTriplets(a_priori_triplets.begin(),
                                  a_priori_triplets.end());
  pattern_penalty.makeCompressed();
  formatted_sparse_multiplication_triple_product(
      pattern_penalty, KCompressed_source_quadborder, WCompressed_border,
      KCompressed_source_quadborder);
  Eigen::SparseMatrix<double> Penalty =
      pattern_penalty.selfadjointView<Eigen::Upper>();
  double mult_time_eigen_penalty = T.toc();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // create Stiffness and Neumann term
  Eigen::SparseMatrix<double> Stiffness(P_sources.cols(), P_sources.cols());
  Stiffness.setZero();
  Eigen::SparseMatrix<double> GradNormal(P_sources.cols(),
                                         P_quad_border.cols());
  GradNormal.setZero();

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
    pattern.setFromTriplets(a_priori_triplets.begin(), a_priori_triplets.end());
    pattern.makeCompressed();
    formatted_sparse_multiplication_triple_product(
        pattern, GradKCompressed, WCompressed, GradKCompressed);
    double mult_time_eigen = T.toc();
    Stiffness += pattern.selfadjointView<Eigen::Upper>();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Neumann
    const usMatrixEvaluator mat_eval_neumann(mom_sources, mom_quad_border,
                                             function);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
        Scomp_neumann;
    std::cout << "Neumann" << std::endl;
    Eigen::SparseMatrix<double> GradKCompressed_neumann =
        createCompressedSparseMatrixUnSymmetric(
            function, mat_eval_neumann, hst_sources, hst_quad_border, eta,
            threshold_gradKernel, P_sources, P_quad_border);

    Eigen::SparseMatrix<double> gradk_n =
        NeumannNormalMultiplication(GradKCompressed_neumann, Normals.row(i));
    gradk_n.makeCompressed();
    GradNormal += gradk_n;
  }
  anz_gradkernel = anz_gradkernel / DIM;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Final Neumann Term
  std::cout << "Grad Neumann done" << std::endl;
  Eigen::SparseMatrix<double> Neumann =
      GradNormal * (WCompressed_border.selfadjointView<Eigen::Upper>() *
                    KCompressed_source_quadborder.transpose())
                       .eval();
  // Nitsche’s Term
  Eigen::SparseMatrix<double> Neumann_Nitsche = Neumann.transpose();
  std::cout << "Neumann done" << std::endl;
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
  std::cout << "FCompressed done" << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // GCompressed = right hand side penalty
  FMCA::Vector GCompressed =
      KCompressed_source_quadborder *
      (WCompressed_border.selfadjointView<Eigen::Upper>() *
       hst_quad_border.sampletTransform(u_bc_reordered))
          .eval();
  std::cout << "GCompressed done" << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // NCompressed= right hand side Nitsche
  FMCA::Vector NCompressed =
      GradNormal * (WCompressed_border.selfadjointView<Eigen::Upper>() *
                    hst_quad_border.sampletTransform(u_bc_reordered))
                       .eval();
  std::cout << "NCompressed done" << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::SparseMatrix<double> Matrix_system_half =
      (Stiffness + beta * Penalty - (Neumann + Neumann_Nitsche));

  Eigen::SparseMatrix<double> Matrix_system =
      Matrix_system_half.selfadjointView<Eigen::Upper>();

  FMCA::Vector Matrix_system_diagonal = Matrix_system.diagonal();
  std::cout << "Min element diagonal:                               "
            << Matrix_system_diagonal.minCoeff() << std::endl;

  std::cout << "Number of element per row system matrix:            "
            << Matrix_system.nonZeros() / Matrix_system.rows() << std::endl;

  int entries_smaller_threshold = countSmallerThan(Matrix_system, 1e-6);
  std::cout << "Number of entries smaller than threshold:            "
            << entries_smaller_threshold << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Solver
  EigenCholesky choleskySolver;
  choleskySolver.compute(Matrix_system);
  T.tic();
  FMCA::Vector u =
      choleskySolver.solve(FCompressed + beta * GCompressed - NCompressed);
  FMCA::Scalar solver_time_chol = T.toc();
  std::cout << "solver time SimplicialLDLT:                                    "
            << solver_time_chol << std::endl;

  std::cout << "residual error SimplicialLDLT:                                 "
            << ((Matrix_system)*u -
                (FCompressed + beta * GCompressed - NCompressed))
                   .norm()
            << std::endl;
  return u;
}

// This funtion is the same as SolvePoisson but the weights are constant, so the weights 
// input are Scalar w_vec and w_vec_border
FMCA::Vector SolvePoisson_constantWeighs(
    const FMCA::Scalar &DIM, FMCA::Matrix &P_sources, FMCA::Matrix &P_quad,
    FMCA::Scalar &w_vec, FMCA::Matrix &P_quad_border,
    FMCA::Scalar &w_vec_border, FMCA::Matrix &Normals, FMCA::Vector &u_bc,
    FMCA::Vector &f, const FMCA::Scalar &sigma, const FMCA::Scalar &eta,
    const FMCA::Index &dtilde, const FMCA::Scalar &threshold_kernel,
    const FMCA::Scalar &threshold_gradKernel,
    const FMCA::Scalar &threshold_weights, const FMCA::Scalar &MPOLE_DEG,
    const FMCA::Scalar &beta, const std::string &kernel_type) {
  FMCA::Tictoc T;
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
    for (const auto &i : triplets) {
      y2(i.row()) += i.value() * x(i.col());
      if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
    }
    y2 = hst_sources.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  std::cout << "compression error =          " << sqrt(err / nrm) << std::endl;

  int anz = triplets.size() / P_sources.cols();
  std::cout << "Anz:                         " << anz << std::endl;

  Eigen::SparseMatrix<double> KCompressed_source(P_sources.cols(),
                                                 P_sources.cols());
  KCompressed_source.setFromTriplets(triplets.begin(), triplets.end());
  KCompressed_source.makeCompressed();
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
  // Penalty: K_{Psources, Pquad_border} * W_border
  // * K_{Psources, Pquad_border}.transpose()
  const FMCA::CovarianceKernel kernel_funtion_sources_quadborder(kernel_type,
                                                                 sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel_sources_quadborder(
      mom_sources, mom_quad_border, kernel_funtion_sources_quadborder);
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
      Kcomp_sources_quadborder;
  std::cout << "Kernel Source-QuadBorder" << std::endl;
  Eigen::SparseMatrix<double> KCompressed_source_quadborder =
      createCompressedSparseMatrixUnSymmetric(
          kernel_funtion_sources_quadborder, mat_eval_kernel_sources_quadborder,
          hst_sources, hst_quad_border, eta, threshold_kernel, P_sources,
          P_quad_border);

  Sparse pattern_Penalty(P_sources.cols(), P_sources.cols());
  pattern_Penalty.setFromTriplets(a_priori_triplets.begin(),
                                  a_priori_triplets.end());
  pattern_Penalty.makeCompressed();

  formatted_sparse_multiplication_dotproduct(pattern_Penalty,
                                             KCompressed_source_quadborder,
                                             KCompressed_source_quadborder);
  pattern_Penalty *= w_vec_border;
  pattern_Penalty.makeCompressed();
  Eigen::SparseMatrix<double> Penalty =
      pattern_Penalty.selfadjointView<Eigen::Upper>();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // create Stiffness and Neumann term
  Eigen::SparseMatrix<double> Stiffness(P_sources.cols(), P_sources.cols());
  Stiffness.setZero();

  Eigen::SparseMatrix<double> GradNormal(P_sources.cols(),
                                         P_quad_border.cols());
  GradNormal.setZero();

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
    pattern.setFromTriplets(a_priori_triplets.begin(), a_priori_triplets.end());
    pattern.makeCompressed();

    formatted_sparse_multiplication_dotproduct(pattern, GradKCompressed,
                                               GradKCompressed);
    pattern *= w_vec;
    double mult_time_eigen = T.toc();
    pattern.makeCompressed();
    Stiffness += pattern.selfadjointView<Eigen::Upper>();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Neumann
    const usMatrixEvaluator mat_eval_neumann(mom_sources, mom_quad_border,
                                             function);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
        Scomp_neumann;
    std::cout << "Neumann" << std::endl;
    Eigen::SparseMatrix<double> GradKCompressed_neumann =
        createCompressedSparseMatrixUnSymmetric(
            function, mat_eval_neumann, hst_sources, hst_quad_border, eta,
            threshold_gradKernel, P_sources, P_quad_border);

    Eigen::SparseMatrix<double> gradk_n =
        NeumannNormalMultiplication(GradKCompressed_neumann, Normals.row(i));
    gradk_n.makeCompressed();
    GradNormal += gradk_n;
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Final Neumann Term
  Sparse pattern_neumann(P_sources.cols(), P_sources.cols());
  pattern_neumann.setFromTriplets(a_priori_triplets.begin(),
                                  a_priori_triplets.end());
  pattern_neumann.makeCompressed();

  formatted_sparse_multiplication_dotproduct(pattern_neumann, GradNormal,
                                             KCompressed_source_quadborder);
  pattern_neumann *= w_vec_border;
  pattern_neumann.makeCompressed();

  Eigen::SparseMatrix<double> Neumann =
      pattern_neumann.selfadjointView<Eigen::Upper>();
  Eigen::SparseMatrix<double> Neumann_Nitsche = Neumann.transpose();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // reorder the boundary conditions and the rhs to follow the samplets order
  FMCA::Vector u_bc_reordered = hst_quad_border.toClusterOrder(u_bc);
  FMCA::Vector f_reordered = hst_quad.toClusterOrder(f);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // FCompressed = right hand side of the problem involving the source term f
  FMCA::Vector FCompressed =
      KCompressed_sourcequad * hst_quad.sampletTransform(f_reordered);
  FCompressed *= w_vec;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // GCompressed = right hand side penalty
  FMCA::Vector GCompressed = KCompressed_source_quadborder *
                             hst_quad_border.sampletTransform(u_bc_reordered);
  GCompressed *= w_vec_border;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // NCompressed= right hand side Nitsche
  FMCA::Vector NCompressed =
      GradNormal * hst_quad_border.sampletTransform(u_bc_reordered);
  NCompressed *= w_vec_border;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::SparseMatrix<double> Matrix_system_half =
      (Stiffness + beta * Penalty - (Neumann + Neumann_Nitsche));

  applyPatternAndFilter(Matrix_system_half, a_priori_triplets, 1e-8);
  Eigen::SparseMatrix<double> Matrix_system =
      Matrix_system_half.selfadjointView<Eigen::Upper>();

  FMCA::Vector Matrix_system_diagonal = Matrix_system.diagonal();
  std::cout << "Min element diagonal:                               "
            << Matrix_system_diagonal.minCoeff() << std::endl;

  std::cout << "Number of element per row system matrix:            "
            << Matrix_system.nonZeros() / Matrix_system.rows() << std::endl;

  int entries_smaller_threshold = countSmallerThan(Matrix_system, 1e-8);
  std::cout << "Number of entries smaller than threshold:            "
            << entries_smaller_threshold << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Solver
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                        Eigen::MetisOrdering<int>>
      choleskySolver;
  choleskySolver.compute(Matrix_system);
  // u = solution in Samplet Basis
  FMCA::Vector u =
      choleskySolver.solve(FCompressed + beta * GCompressed - NCompressed);
  std::cout << "residual error:                                  "
            << ((Matrix_system)*u -
                (FCompressed + beta * GCompressed - NCompressed))
                   .norm()
            << std::endl;
  return u;
}