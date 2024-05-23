#include <cstdlib>
#include <iostream>
//##############################
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

double computeTrace(const Eigen::SparseMatrix<double> &mat) {
  double trace = 0.0;
  // Iterate only over the diagonal elements
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
      if (it.row() == it.col()) {  // Check if it is a diagonal element
        trace += it.value();
      }
    }
  }
  return trace;
}

int countSmallerThan(const Eigen::SparseMatrix<double> &matrix,
                     FMCA::Scalar threshold) {
  int count = 0;
  // Iterate over all non-zero elements.
  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it;
         ++it) {
      if (it.value() < threshold) {
        count++;
      }
    }
  }
  return count / matrix.rows();
}

bool isSymmetric(const Eigen::SparseMatrix<double> &matrix,
                 FMCA::Scalar tol = 1e-10) {
  if (matrix.rows() != matrix.cols())
    return false;  // Non-square matrices are not symmetric

  // Iterate over the outer dimension
  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it;
         ++it) {
      if (std::fabs(it.value() - matrix.coeff(it.col(), it.row())) > tol)
        return false;
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<FMCA::Scalar> runSimulation(FMCA::Scalar sigma, string kernel_type) {
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
  const FMCA::Scalar threshold_kernel = 1e-3;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 4;
  // const FMCA::Scalar sigma = 2 * 1 / (sqrt(P_sources.cols()));
  // const FMCA::Scalar threshold_gradKernel = 3*threshold_kernel/sigma;
  const FMCA::Scalar beta = 1000;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // FMCA::Scalar threshold_gradKernel = 3 * threshold_kernel / sigma;
  FMCA::Scalar threshold_gradKernel = threshold_kernel / (sigma * sigma);
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
  std::cout << "MPOLE_DEG =                                       " << MPOLE_DEG
            << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Kernel source-source compression for the pattern of the multiplication
  // and the final solution
  const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
  const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources,
                                                 kernel_funtion_ss);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
  Kcomp_ss.init(hst_sources, eta, threshold_kernel);
  Kcomp_ss.compress(mat_eval_kernel_ss);
  const auto &trips_ss = Kcomp_ss.triplets();
  FMCA::Scalar anz_kernel = trips_ss.size() / P_sources.cols(); 
  std::cout << "anz kernel ss:                                      "
            << anz_kernel << std::endl;
  Eigen::SparseMatrix<double> Kcomp_ss_Sparse(P_sources.cols(),
                                              P_sources.cols());
  Kcomp_ss_Sparse.setFromTriplets(trips_ss.begin(), trips_ss.end());
  Kcomp_ss_Sparse.makeCompressed();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Weights compression
  FMCA::Vector w_perm = hst_quad.toClusterOrder(w_vec);
  FMCA::SparseMatrix<FMCA::Scalar> W(w_perm.size(), w_perm.size());
  for (int i = 0; i < w_perm.size(); ++i) {
    W.insert(i, i) = w_perm(i);
  }
  FMCA::SparseMatrixEvaluator mat_eval_weights(W);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp;
  Wcomp.init(hst_quad, eta, threshold_weights);
  T.tic();
  Wcomp.compress(mat_eval_weights);
  double compressor_time_weights = T.toc();
  const auto &trips_weights = Wcomp.triplets();
  std::cout << "anz weights:                                      "
            << trips_weights.size() / P_quad.cols() << std::endl;
  Eigen::SparseMatrix<double> Wcomp_Sparse(P_quad.cols(), P_quad.cols());
  Wcomp_Sparse.setFromTriplets(trips_weights.begin(), trips_weights.end());
  Wcomp_Sparse.makeCompressed();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Weights border compression
  FMCA::Vector w_perm_border = hst_quad_border.toClusterOrder(w_vec_border);
  // FMCA::Vector w_perm_border =
  // w_vec_border(permutationVector(hst_quad_border));
  FMCA::SparseMatrix<FMCA::Scalar> W_border(w_perm_border.size(),
                                            w_perm_border.size());
  for (int i = 0; i < w_perm_border.size(); ++i) {
    W_border.insert(i, i) = w_perm_border(i);
  }
  FMCA::SparseMatrixEvaluator mat_eval_weights_border(W_border);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp_border;
  Wcomp_border.init(hst_quad_border, eta, threshold_weights);
  Wcomp_border.compress(mat_eval_weights_border);
  const auto &trips_weights_border = Wcomp_border.triplets();
  std::cout << "anz weights border:                               "
            << trips_weights_border.size() / P_quad_border.cols() << std::endl;
  Eigen::SparseMatrix<double> Wcomp_border_Sparse(P_quad_border.cols(),
                                                  P_quad_border.cols());
  Wcomp_border_Sparse.setFromTriplets(trips_weights_border.begin(),
                                      trips_weights_border.end());
  Wcomp_border_Sparse.makeCompressed();
  std::cout << std::string(80, '-') << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Kernel (sources,quad) compression
  const FMCA::CovarianceKernel kernel_funtion(kernel_type, sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel(mom_sources, mom_quad,
                                                kernel_funtion);
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
  Kcomp.init(hst_sources, hst_quad, eta, threshold_kernel);
  T.tic();
  Kcomp.compress(mat_eval_kernel);
  double compressor_time = T.toc();
  std::cout << "compression time kernel :                         "
            << compressor_time << std::endl;
  const auto &trips = Kcomp.triplets();
  std::cout << "anz kernel:                                       "
            << trips.size() / P_sources.cols() << std::endl;
  Eigen::SparseMatrix<double> Kcomp_Sparse(P_sources.cols(), P_quad.cols());
  Kcomp_Sparse.setFromTriplets(trips.begin(), trips.end());
  Kcomp_Sparse.makeCompressed();
  std::cout << std::string(80, '-') << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Penalty
  // construct the matrix M_b = K_{Psources, Pquad_border} * W_border *
  // K_{Psources, Pquad_border}.transpose()
  const FMCA::CovarianceKernel kernel_funtion_sources_quadborder(kernel_type,
                                                                 sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel_sources_quadborder(
      mom_sources, mom_quad_border, kernel_funtion_sources_quadborder);
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
      Kcomp_sources_quadborder;
  Kcomp_sources_quadborder.init(hst_sources, hst_quad_border, eta,
                                threshold_kernel);
  T.tic();
  Kcomp_sources_quadborder.compress(mat_eval_kernel_sources_quadborder);
  double compressor_time_sources_quadborder = T.toc();
  std::cout << "compression time kernel_sources_quadborder :      "
            << compressor_time_sources_quadborder << std::endl;
  const auto &trips_sources_quadborder = Kcomp_sources_quadborder.triplets();
  std::cout << "anz kernel_sources_quadborder :                         "
            << trips_sources_quadborder.size() / P_sources.cols() << std::endl;
  Eigen::SparseMatrix<double> Kcomp_sources_quadborder_Sparse(
      P_sources.cols(), P_quad_border.cols());
  Kcomp_sources_quadborder_Sparse.setFromTriplets(
      trips_sources_quadborder.begin(), trips_sources_quadborder.end());
  Kcomp_sources_quadborder_Sparse.makeCompressed();
  std::cout << std::string(80, '-') << std::endl;
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
    Scomp.init(hst_sources, hst_quad, eta, threshold_gradKernel);
    T.tic();
    Scomp.compress(mat_eval);
    double compressor_time = T.toc();
    std::cout << "compression time gradKernel " << i << "                      "
              << compressor_time << std::endl;
    const auto &trips = Scomp.triplets();
    std::cout << "anz gradKernel:                                   "
              << trips.size() / P_sources.cols() << std::endl;
    anz_gradkernel += (trips.size() / P_sources.cols());
    Eigen::SparseMatrix<double> Scomp_Sparse(P_sources.cols(), P_quad.cols());
    Scomp_Sparse.setFromTriplets(trips.begin(), trips.end());
    Scomp_Sparse.makeCompressed();

    T.tic();
    Eigen::SparseMatrix<double> gradk =
        Scomp_Sparse * (Wcomp_Sparse * Scomp_Sparse.transpose()).eval();
    double mult_time_eigen = T.toc();
    std::cout << "eigen mult time component " << i << "                        "
              << mult_time_eigen << std::endl;
    gradk.makeCompressed();
    stiffness += gradk.selfadjointView<Eigen::Upper>();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Neumann
    const usMatrixEvaluator mat_eval_neumann(mom_sources, mom_quad_border,
                                             function);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
        Scomp_neumann;
    Scomp_neumann.init(hst_sources, hst_quad_border, eta, threshold_gradKernel);
    Scomp_neumann.compress(mat_eval_neumann);
    const auto &trips_neumann = Scomp_neumann.triplets();
    std::cout << "anz gradKernel*n:                                   "
              << trips_neumann.size() / P_sources.cols() << std::endl;
    Eigen::SparseMatrix<double> Scomp_Sparse_neumann(P_sources.cols(),
                                                     P_quad_border.cols());
    Scomp_Sparse_neumann.setFromTriplets(trips_neumann.begin(),
                                         trips_neumann.end());
    Scomp_Sparse_neumann.makeCompressed();
    Eigen::SparseMatrix<double> gradk_n =
        NeumannNormalMultiplication(Scomp_Sparse_neumann, Normals.row(i));
    gradk_n.makeCompressed();
    grad_times_normal += gradk_n;
  }
  anz_gradkernel = anz_gradkernel/DIM; 
  std::cout << std::string(80, '-') << std::endl;

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
  Eigen::SparseMatrix<double> Matrix_system =
      stiffness + beta * Mb - (Neumann + Neumann_Nitsche);
  // - (Neumann + Neumann_Nitsche)
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
  // - N_comp
  FMCA::Vector u_natural_order = hst_sources.toNaturalOrder(u);
  FMCA::IO::print2m(
      "results_matlab/solution_square_penalty_compressed_SampletBasis.m",
      "sol_square_penalty_compressed_SampletBasis", u_natural_order, "w");
  // Check the solver
  std::cout << "residual error:                                  "
            << ((Matrix_system)*u - (F_comp + beta * G_comp - N_comp)).norm()
            << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::vector<const H2SampletTree *> adaptive_tree =
      adaptiveTreeSearch(hst_sources, u, 1e-4 * u.squaredNorm());
  const FMCA::Index nclusters =
      std::distance(hst_sources.begin(), hst_sources.end());

  FMCA::Vector thres_tdata = u;
  thres_tdata.setZero();
  FMCA::Index nnz = 0;
  for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    if (adaptive_tree[i] != nullptr) {
      const H2SampletTree &node = *(adaptive_tree[i]);
      const FMCA::Index ndist =
          node.is_root() ? node.Q().cols() : node.nsamplets();
      thres_tdata.segment(node.start_index(), ndist) =
          u.segment(node.start_index(), ndist);
      nnz += ndist;
    }
  }
  std::cout << "active coefficients: " << nnz << " / " << P_sources.cols()
            << std::endl;
  std::cout << "tree error: " << (thres_tdata - u).norm() / u.norm()
            << std::endl;
  /*
  // plot the active leaves bounding boxes
  const FMCA::Scalar max_coeff = u.cwiseAbs().maxCoeff();
  std::cout << "max_coeff:      " << max_coeff << std::endl;
  const FMCA::Index ncluster =
      std::distance(hst_sources.begin(), hst_sources.end());
  std::cout << "ncluster:      " << ncluster << std::endl;
  std::vector<bool> active(ncluster);
  FMCA::markActiveClusters(active, hst_sources, u, max_coeff, 0.01 *
  max_coeff); std::vector<const H2SampletTree *> active_leafs;
  FMCA::getActiveLeafs(active_leafs, active, hst_sources);

  std::vector<FMCA::Matrix> bbvec_active;
  for (const auto *leaf : active_leafs) {
    if (leaf != nullptr) {
      bbvec_active.push_back(leaf->bb());
    }
  }
  FMCA::IO::plotBoxes2D("active_boxes_Poisson_square.vtk", bbvec_active);
  */
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::SparseMatrix<double> Kcomp_ss_Sparse_symm =
      Kcomp_ss_Sparse.selfadjointView<Eigen::Upper>();
  FMCA::Vector KU = Kcomp_ss_Sparse_symm * u;
  FMCA::Vector KU_transformed = hst_sources.inverseSampletTransform(KU);
  FMCA::Vector KU_permuted = hst_sources.toNaturalOrder(KU_transformed);
  // KU_transformed(inversePermutationVector(hst_sources));
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // analytical sol of the problem
  FMCA::Vector analytical_sol(P_sources.cols());
  for (int i = 0; i < P_sources.cols(); ++i) {
    double x = P_sources(0, i);
    double y = P_sources(1, i);
    analytical_sol[i] = sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }
  FMCA::Scalar error = (KU_permuted - analytical_sol).norm() /
                       analytical_sol.norm();

  // FMCA::Vector absolute_error(P_sources.cols());
  // for (int i = 0; i < P_sources.cols(); ++i) {
  //   absolute_error[i] = abs(KU_permuted[i] - analytical_sol[i]);
  // }
  // FMCA::Vector relative_error(P_sources.cols());
  // for (int i = 0; i < P_sources.cols(); ++i) {
  //   relative_error[i] = abs(KU_permuted[i] - analytical_sol[i]) /
  //   abs(analytical_sol[i]);
  // }

  // FMCA::IO::print2m("results_matlab/solution_square_penalty_compressed.m",
  //                   "sol_square_penalty_compressed", KU_permuted, "w");
  // FMCA::IO::print2m("error_square_penalty_absolute_compressed.m",
  //                   "absolute_error_penalty_compressed", absolute_error,
  //                   "w");
  // FMCA::IO::print2m("error_square_penalty_relative_compressed.m",
  // "relative_error_penalty_compressed",
  //                   relative_error, "w");

  return {error, anz_kernel, anz_gradkernel};
}


int main() {
    vector<double> sigmas = {0.05, 0.03, 0.01, 0.008};
    vector<string> kernel_types = {"MATERN32", "GAUSSIAN"};

    ofstream outFile("simulation_results_poisson_square.txt");
    outFile << "Sigma" << "& Kernel_type" << "& Error" << "& anz_kernel" << "& anz_gradKernel \\ "<< endl;
    for (const string& kernel_type : kernel_types) {
        for (double sigma : sigmas) {
                    try {
                        // Call the simulation function
                        std::vector<FMCA::Scalar> result = runSimulation(sigma, kernel_type);
                        // Write the parameters and error to the output file
                        outFile << sigma << "&" << kernel_type << "&" << result[0] << "&" << result[1] << "&" << result[2] << "\\"
                                << endl;
                    } catch (const std::runtime_error& e) {
                        // Log error details and proceed with next parameters
                        outFile << sigma
                                << "&" << kernel_type << " Error: Decomposition failed"
                                << endl;
                        cout << "Error during simulation: " << e.what() << endl;
                    } catch (const std::exception& e) {
                        // General exception log, for other types of failures
                        outFile << sigma
                                << "&" << kernel_type << ", Error: " << e.what()
                                << endl;
                    }
                }
            }

    outFile.close();
    return 0;
}