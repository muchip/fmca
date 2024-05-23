#include <cstdlib>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
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

int countSmallerThan(const Eigen::SparseMatrix<double>& matrix, FMCA::Scalar threshold) {
    int count = 0;
    // Iterate over all non-zero elements.
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
            if (it.value() < threshold) {
                count++;
            }
        }
    }
    return count/matrix.rows();
}

void setZeroSmallerThan(Eigen::SparseMatrix<double>& matrix, double threshold) {
    // Iterate over all non-zero elements directly modifying the original matrix
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
            if (it.value() < threshold) {
                // Directly modify the element
                const_cast<double&>(it.valueRef()) = 0;
            }
        }
    }
    // Prune the matrix to remove the newly zeroed elements
    matrix.prune([](int i, int j, double value) {
        return value != 0;
    });
}

bool isSymmetric(const Eigen::SparseMatrix<double>& matrix, FMCA::Scalar tol = 1e-10) {
    if (matrix.rows() != matrix.cols())
        return false;  // Non-square matrices are not symmetric

    // Iterate over the outer dimension
    for (int k=0; k < matrix.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<double>::InnerIterator it(matrix,k); it; ++it) {
            if (std::fabs(it.value() - matrix.coeff(it.col(), it.row())) > tol)
                return false;
        }
    }
    return true;
}



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
  readTXT("data/vertices_square_01.txt", P_sources, 2);
  readTXT("data/quadrature7_points_square_01.txt", P_quad, 2);
  readTXT("data/quadrature7_weights_square_01.txt", w_vec);
  readTXT("data/quadrature_border_01.txt", P_quad_border, 2);
  readTXT("data/weights_border_01.txt", w_vec_border);
  readTXT("data/normals_01.txt", Normals, 2);

  std::cout << "P_sources dim  " << P_sources.rows() << "," << P_sources.cols()
            << std::endl;
  std::cout << "P_quad dim  " << P_quad.rows() << "," << P_quad.cols()
            << std::endl;
  std::cout << "w_vec dim  " << w_vec.rows() << "," << w_vec.cols()
            << std::endl;
  std::cout << "P_quad_border dim  " << P_quad_border.rows() << ","
            << P_quad_border.cols() << std::endl;
  std::cout << "w_vec_border dim  " << w_vec_border.rows() << ","
            << w_vec_border.cols() << std::endl;
  std::cout << "Normals dim  " << Normals.rows() << "," << Normals.cols()
            << std::endl;
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
  const FMCA::Index dtilde = 4;
  const FMCA::Scalar threshold_kernel = 1e-4;
  const FMCA::Scalar threshold_gradKernel = 1e-0;
  const FMCA::Scalar threshold_weights = 0;
  const FMCA::Scalar MPOLE_DEG = 6;
  const FMCA::Scalar sigma = 0.01;
  const FMCA::Scalar beta = 1000;
  std::string kernel_type = "EXPONENTIAL";
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
  std::cout << "threshold_kernel =                                " << threshold_kernel
            << std::endl;
  std::cout << "threshold_gradKernel =                            " << threshold_gradKernel
            << std::endl;
  std::cout << "threshold_weights =                               "
            << threshold_weights << std::endl;
  std::cout << "MPOLE_DEG =                                       " << MPOLE_DEG
            << std::endl;
  std::cout << std::string(80, '-') << std::endl;
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
  Kcomp_sources_quadborder.init(hst_sources, hst_quad_border, eta, threshold_kernel);
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
  T.tic();
  Eigen::SparseMatrix<double> Mb =
      Kcomp_sources_quadborder_Sparse *
      (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
       Kcomp_sources_quadborder_Sparse.transpose())
          .eval();
  double mult_time_eigen_Mb = T.toc();
  std::cout << "eigen multiplication time matrix Mb:              "
            << mult_time_eigen_Mb << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // create stiffness and Neumann term
  Eigen::SparseMatrix<double> stiffness(P_sources.cols(), P_sources.cols());
  stiffness.setZero();

  Eigen::SparseMatrix<double> grad_times_normal(P_sources.cols(),
                                                P_quad_border.cols());
  grad_times_normal.setZero();

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
    Eigen::SparseMatrix<double> Scomp_Sparse(P_sources.cols(), P_quad.cols());
    Scomp_Sparse.setFromTriplets(trips.begin(), trips.end());
    Scomp_Sparse.makeCompressed();

    T.tic();
    Eigen::SparseMatrix<double> gradk =
        Scomp_Sparse * (Wcomp_Sparse * Scomp_Sparse.transpose()).eval();
    double mult_time_eigen = T.toc();
    std::cout << "eigen mult time component " << i << "                        "
              << mult_time_eigen << std::endl;
    std::cout << "NON ZEROS GRAD K(S,S)        " << gradk.nonZeros()
              << std::endl;
    gradk.makeCompressed();
    stiffness += gradk.selfadjointView<Eigen::Upper>();
    FMCA::IO::print2m("results_matlab/Scomp_Sparse_poisson_square.m",
                    "Scomp_Sparse", Scomp_Sparse, "w");
    FMCA::IO::print2m("results_matlab/Wcomp_Sparse_poisson_square.m",
                    "Wcomp_Sparse", Wcomp_Sparse, "w");    
    FMCA::IO::print2m("results_matlab/gradk_poisson_square.m",
                    "gradk", gradk, "w");          
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Neumann
    const usMatrixEvaluator mat_eval_neumann(mom_sources, mom_quad_border,
                                             function);
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
        Scomp_neumann;
    Scomp_neumann.init(hst_sources, hst_quad_border, eta, threshold_kernel);
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
  FMCA::Vector u_bc_reordered = hst_quad_border.toClusterOrder(u_bc);
  FMCA::Vector f_reordered = hst_quad.toClusterOrder(f);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  FMCA::Vector F_comp = Kcomp_Sparse * (Wcomp_Sparse.selfadjointView<Eigen::Upper>() *
                      hst_quad.sampletTransform(f_reordered))
                         .eval();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  FMCA::Vector G_comp = Kcomp_sources_quadborder_Sparse *
                        (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
                         hst_quad_border.sampletTransform(u_bc_reordered)).eval();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  FMCA::Vector N_comp =
      grad_times_normal * (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
                           hst_quad_border.sampletTransform(u_bc_reordered)).eval();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::SparseMatrix<double> Matrix_system = stiffness - (Neumann + Neumann_Nitsche) + beta * Mb;
  FMCA::IO::print2m("results_matlab/Stiffness_poisson_square.m",
                    "Stiffness", stiffness, "w");  
  FMCA::IO::print2m("results_matlab/Neumann_poisson_square.m",
                    "Neumann", Neumann, "w");
  FMCA::IO::print2m("results_matlab/Penalty_poisson_square.m",
                    "Penalty", Mb, "w");
  FMCA::IO::print2m("results_matlab/Matrix_system_poisson_square.m",
                    "Matrix_system", Matrix_system, "w");
  // - (Neumann + Neumann_Nitsche)
  // Trace
  double trace = computeTrace(Matrix_system);
  std::cout << "Trace of the system matrix:                         " << trace << std::endl;
  std::cout << "Number of element per row system matrix:            " << Matrix_system.nonZeros()/Matrix_system.rows() << std::endl;
  // Check elements  
  int entries_smaller_threshold =  countSmallerThan(Matrix_system, 1e-6);
  std::cout << "Number of entries smaller than threshold:            " << entries_smaller_threshold << std::endl;
  // check Symmetry of Matrix_system
  bool issymmetric = isSymmetric(Matrix_system);
  std::cout << "Matrix of the system is symmetric:       " << issymmetric << std::endl;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Solver
  
  // Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
  // solver.compute(Matrix_system);
  // FMCA::Vector u = solver.solve(F_comp + beta * G_comp);

  EigenCholesky choleskySolver;
  choleskySolver.compute(Matrix_system); 
  std::cout << "Matrix of the system is PD:              " << (choleskySolver.info() == Eigen::Success) << std::endl;
  FMCA::Vector u = choleskySolver.solve(F_comp + beta * G_comp - N_comp);

 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Evaluation of the numerical solution
  const FMCA::CovarianceKernel kernel_funtion_ss(kernel_type, sigma);
  const MatrixEvaluatorKernel mat_eval_kernel_ss(mom_sources, kernel_funtion_ss);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Kcomp_ss;
  Kcomp_ss.init(hst_sources, eta, threshold_kernel);
  Kcomp_ss.compress(mat_eval_kernel_ss);
  const auto &trips_ss = Kcomp_ss.triplets();
  std::cout << "anz kernel ss:                                      "
            << trips_ss.size() / P_sources.cols() << std::endl;
  Eigen::SparseMatrix<double> Kcomp_ss_Sparse(P_sources.cols(),
                                              P_sources.cols());
  Kcomp_ss_Sparse.setFromTriplets(trips_ss.begin(), trips_ss.end());
  Kcomp_ss_Sparse.makeCompressed();
  Eigen::SparseMatrix<double> Kcomp_ss_Sparse_symm =
      Kcomp_ss_Sparse.selfadjointView<Eigen::Upper>();
  std::cout << "NON ZEROS K(S,S)        " << Kcomp_ss_Sparse_symm.nonZeros()
            << std::endl;
  return 0;
}
