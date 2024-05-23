// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#include <cstdlib>
#include <iostream>
// #include
// </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/MetisSupport>
#include </opt/homebrew/Cellar/metis/5.1.0/include/metis.h>

#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/MetisSupport>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Sparse>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/SparseCholesky>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/SparseQR>
#include <Eigen/OrderingMethods>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/FormattedMultiplication/FormattedMultiplication.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "Uzawa.h"
#include "read_files_txt.h"

#define DIM 2
#define GRADCOMPONENT 0

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluatorKernel =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using EigenCholesky =
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                         Eigen::MetisOrdering<int>>;

int main() {
  // Initialize the matrices of source points, quadrature points, weights
  FMCA::Tictoc T;
  int NPTS_SOURCE;
  int NPTS_SOURCE_BORDER;
  int NPTS_QUAD;
  int NPTS_QUAD_BORDER;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_quad_border;
  //////////////////////////////////////////////////////////////////////////////
  readTXT("data/rect_vertices.txt", P_sources, NPTS_SOURCE, 2);
  readTXT("data/rect_quad.txt", P_quad, NPTS_QUAD, 2);
  readTXT("data/rectangle_quad_border.txt", P_quad_border, NPTS_QUAD_BORDER, 2);
  int N_WEIGHTS = NPTS_QUAD;
  FMCA::Vector w_vec(NPTS_QUAD);
  w_vec.setOnes();
  w_vec = w_vec*1/4;
  FMCA::Vector w_vec_border(NPTS_QUAD_BORDER);
  w_vec_border.setOnes();
  w_vec_border = w_vec_border*1/2;
  // //////////////////////////////////////////////////////////////////////////////
  // Boundary conditions
  FMCA::Vector indices_x0;
  FMCA::Vector indices_x10;
  for (int i = 0; i < P_sources.cols(); ++i) {
    if (P_sources(0, i) == 0) {
      indices_x0.conservativeResize(indices_x0.size() + 1);
      indices_x0(indices_x0.size() - 1) = i;
    } else if (P_sources(0, i) == 10) {
      indices_x10.conservativeResize(indices_x10.size() + 1);
      indices_x10(indices_x10.size() - 1) = i;
    }
  }
  // Boundary sources points
  NPTS_SOURCE_BORDER = indices_x0.size() + indices_x10.size();
  FMCA::Matrix P_sources_border(2, NPTS_SOURCE_BORDER);
  for (int i = 0; i < indices_x0.size(); ++i) {
    P_sources_border.col(i) = P_sources.col(indices_x0(i));
  }
  for (int i = 0; i < indices_x10.size(); ++i) {
    P_sources_border.col(i + indices_x0.size()) = P_sources.col(indices_x10(i));
  }
  // U_BC vector
  FMCA::Vector u_bc(NPTS_QUAD_BORDER);
  for (int i = 0; i < indices_x0.size() - 1; ++i) {
    u_bc[i] = 1;
  }
  for (int i = indices_x0.size() - 1; i < NPTS_QUAD_BORDER; ++i) {
    u_bc[i] = 1001;
  }
  //////////////////////////////////////////////////////////////////////////////
  // Right hand side f = 6 * x
  FMCA::Vector f;
  f.resize(NPTS_QUAD);
  for (int i = 0; i < NPTS_QUAD; ++i) {
    f(i) = P_quad(0, i) * 6;
  }
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 6;
  const FMCA::Scalar threshold = 1e-6;
  const FMCA::Scalar MPOLE_DEG = 6;
  const FMCA::Scalar sigma = 1;
  const FMCA::Scalar beta = 1000;
  //////////////////////////////////////////////////////////////////////////////
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const Moments mom_quad(P_quad, MPOLE_DEG);
  const Moments mom_sources_border(P_sources_border, MPOLE_DEG);
  const Moments mom_quad_border(P_quad_border, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
  const SampletMoments samp_mom_sources_border(P_sources_border, dtilde - 1);
  const SampletMoments samp_mom_quad_border(P_quad_border, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
  H2SampletTree hst_sources_border(mom_sources_border, samp_mom_sources_border,
                                   0, P_sources_border);
  H2SampletTree hst_quad_border(mom_quad_border, samp_mom_quad_border, 0,
                                P_quad_border);
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
  std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
  std::cout << "Number of weights = " << N_WEIGHTS << std::endl;
  std::cout << "minimum element:                     "
            << *std::min_element(w_vec.begin(), w_vec.end()) << std::endl;
  std::cout << "maximum element:                     "
            << *std::max_element(w_vec.begin(), w_vec.end()) << std::endl;
  std::cout << "Number of sources points border = " << NPTS_SOURCE_BORDER
            << std::endl;
  std::cout << "Number of quad points border = " << NPTS_QUAD_BORDER
            << std::endl;
  std::cout << "Number of weights border = " << NPTS_QUAD_BORDER << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "sigma = " << sigma << std::endl;
  std::cout << "eta = " << eta << std::endl;
  std::cout << "dtilde = " << dtilde << std::endl;
  std::cout << "threshold = " << threshold << std::endl;
  std::cout << "MPOLE_DEG = " << MPOLE_DEG << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  FMCA::Vector u_bc_reordered = u_bc(permutationVector(hst_quad_border));
  FMCA::Vector f_reordered = f(permutationVector(hst_quad));

  //////////////////////////////////////////////////////////////////////////////

  // Weights matrix
  FMCA::Vector w_perm = w_vec(permutationVector(hst_quad));
  FMCA::SparseMatrix<FMCA::Scalar> W(w_perm.size(), w_perm.size());
  for (int i = 0; i < w_perm.size(); ++i) {
    W.insert(i, i) = w_perm(i);
  }  // this foor loop has been checked
  FMCA::SparseMatrixEvaluator mat_eval_weights(W);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp;
  Wcomp.init(hst_quad, eta, threshold);
  T.tic();
  Wcomp.compress(mat_eval_weights);
  double compressor_time_weights = T.toc();
  std::cout << "compression time weights:         " << compressor_time_weights
            << std::endl;
  const auto &trips_weights = Wcomp.triplets();
  std::cout << "anz:                                      "
            << trips_weights.size() / NPTS_QUAD << std::endl;
  Eigen::SparseMatrix<double> Wcomp_Sparse(NPTS_QUAD, NPTS_QUAD);
  Wcomp_Sparse.setFromTriplets(trips_weights.begin(), trips_weights.end());
  Wcomp_Sparse.makeCompressed();
  //////////////////////////////////////////////////////////////////////////////
  // Weights matrix border
  FMCA::Vector w_perm_border = w_vec_border(permutationVector(hst_quad_border));
  FMCA::SparseMatrix<FMCA::Scalar> W_border(w_perm_border.size(),
                                            w_perm_border.size());
  for (int i = 0; i < w_perm_border.size(); ++i) {
    W_border.insert(i, i) = w_perm_border(i);
  }  // this foor loop has been checked
  FMCA::SparseMatrixEvaluator mat_eval_weights_border(W_border);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp_border;
  Wcomp_border.init(hst_quad_border, eta, threshold);
  T.tic();
  Wcomp_border.compress(mat_eval_weights_border);
  double compressor_time_weights_border = T.toc();
  std::cout << "compression time weights_border:         "
            << compressor_time_weights_border << std::endl;
  const auto &trips_weights_border = Wcomp_border.triplets();
  std::cout << "anz:                                      "
            << trips_weights_border.size() / NPTS_QUAD_BORDER << std::endl;
  Eigen::SparseMatrix<double> Wcomp_border_Sparse(NPTS_QUAD_BORDER,
                                                  NPTS_QUAD_BORDER);
  Wcomp_border_Sparse.setFromTriplets(trips_weights_border.begin(),
                                      trips_weights_border.end());
  Wcomp_border_Sparse.makeCompressed();
  //////////////////////////////////////////////////////////////////////////////
  const FMCA::CovarianceKernel kernel_funtion("MATERN32", sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel(mom_sources, mom_quad,
                                                kernel_funtion);
  // Kernel compression
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
  Kcomp.init(hst_sources, hst_quad, eta, threshold);
  T.tic();
  Kcomp.compress(mat_eval_kernel);
  double compressor_time = T.toc();
  std::cout << "compression time kernel :         " << compressor_time
            << std::endl;
  const auto &trips = Kcomp.triplets();
  std::cout << "anz:                                      "
            << trips.size() / NPTS_SOURCE << std::endl;
  Eigen::SparseMatrix<double> Kcomp_Sparse(NPTS_SOURCE, NPTS_QUAD);
  Kcomp_Sparse.setFromTriplets(trips.begin(), trips.end());
  Kcomp_Sparse.makeCompressed();
  std::cout << "determinant of the kernel" << Eigen::MatrixXd(Kcomp_Sparse*Kcomp_Sparse.transpose()).determinant() << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // construct the matrix M_b = K_{Psources, Pquad_border} * W_border *
  // K_{Psources, Pquad_border}.transpose()
  // construct K_{Psources, Pquad_border}
  const FMCA::CovarianceKernel kernel_funtion_sources_quadborder("MATERN32",
                                                                 sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel_sources_quadborder(
      mom_sources, mom_quad_border, kernel_funtion_sources_quadborder);
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree>
      Kcomp_sources_quadborder;
  Kcomp_sources_quadborder.init(hst_sources, hst_quad_border, eta, threshold);
  T.tic();
  Kcomp_sources_quadborder.compress(mat_eval_kernel_sources_quadborder);
  double compressor_time_sources_quadborder = T.toc();
  std::cout << "compression time kernel_sources_quadborder :         "
            << compressor_time_sources_quadborder << std::endl;
  const auto &trips_sources_quadborder = Kcomp_sources_quadborder.triplets();
  std::cout << "anz:                                      "
            << trips_sources_quadborder.size() / NPTS_SOURCE << std::endl;
  Eigen::SparseMatrix<double> Kcomp_sources_quadborder_Sparse(NPTS_SOURCE,
                                                              NPTS_QUAD_BORDER);
  Kcomp_sources_quadborder_Sparse.setFromTriplets(
      trips_sources_quadborder.begin(), trips_sources_quadborder.end());
  Kcomp_sources_quadborder_Sparse.makeCompressed();

  //////////////////////////////////////////////////////////////////////////////
  // create M_b
  T.tic();
  Eigen::SparseMatrix<double> Mb =
      Kcomp_sources_quadborder_Sparse *
      (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
       Kcomp_sources_quadborder_Sparse.transpose())
          .eval();
  double mult_time_eigen_Mb = T.toc();
  std::cout << "eigen multiplication time matrix A:    " << mult_time_eigen_Mb
            << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  Eigen::SparseMatrix<double> stiffness(NPTS_SOURCE, NPTS_SOURCE);
  stiffness.setZero();
  // std::vector<Eigen::Triplet<double>> triplets;
  // for (long long int i = 0; i < NPTS_SOURCE; ++i) {
  //   for (long long int j = 0; j < NPTS_SOURCE; ++j) {
  //     triplets.push_back(Eigen::Triplet<double>(i, j, 0.0));
  //   }
  // }
  // stiffness.setFromTriplets(triplets.begin(), triplets.end());
  // stiffness.makeCompressed();

  for (int i = 0; i < DIM; ++i) {
    std::cout << std::string(80, '-') << std::endl;
    const FMCA::GradKernel function("MATERN32", sigma, 1, i);
    const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);
    //////////////////////////////////////////////////////////////////////////////
    // gradKernel compression
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
    Scomp.init(hst_sources, hst_quad, eta, threshold);
    T.tic();
    Scomp.compress(mat_eval);
    double compressor_time = T.toc();
    std::cout << "compression time component " << i << ":         "
              << compressor_time << std::endl;
    const auto &trips = Scomp.triplets();
    std::cout << "anz:                                      "
              << trips.size() / NPTS_SOURCE << std::endl;
    Eigen::SparseMatrix<double> Scomp_Sparse(NPTS_SOURCE, NPTS_QUAD);
    Scomp_Sparse.setFromTriplets(trips.begin(), trips.end());
    Scomp_Sparse.makeCompressed();
    //std::cout << std::setprecision(2);
    //std::cout << "kernel             " << Eigen::MatrixXd(gradk) << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // eigen multiplication
  T.tic();
  Eigen::SparseMatrix<double> gradk =
      Scomp_Sparse *
      (Wcomp_Sparse.selfadjointView<Eigen::Upper>() * Scomp_Sparse.transpose())
          .eval();
  double mult_time_eigen = T.toc();
  std::cout << "eigen multiplication time component " << "i   ="<< mult_time_eigen
            << std::endl;
  stiffness += gradk;
  }
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  // F_comp = right hand side 1
  FMCA::Vector F_comp =
      Kcomp_Sparse * (Wcomp_Sparse.selfadjointView<Eigen::Upper>() *
                      hst_quad.sampletTransform(6 * f_reordered))
                         .eval();
  std::cout << "F_comp created" << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // G_comp = right hand side 2
  FMCA::Vector G_comp = Kcomp_sources_quadborder_Sparse *
                        (Wcomp_border_Sparse.selfadjointView<Eigen::Upper>() *
                         hst_quad_border.sampletTransform(u_bc_reordered))
                            .eval();
  std::cout << "G_comp created" << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  std::cout << "Frobenius norm stiffness: " << stiffness.norm() << std::endl;
  std::cout << "Frobenius norm A: " << Mb.norm() << std::endl;
  std::cout << "determinant: "
            << Eigen::MatrixXd(stiffness + beta * Mb).determinant() << std::endl;
  // Eigen::LLT<Eigen::MatrixXd> lltOfA(
  //     stiffness + beta * Mb);  // compute the Cholesky decomposition of A
  // if (lltOfA.info() == Eigen::NumericalIssue) {
  //   throw std::runtime_error("Possibly non semi-positive definitie matrix!");
  // }
  //////////////////////////////////////////////////////////////////////////////
  // // solve
  EigenCholesky choleskySolver;
  choleskySolver.compute(stiffness + beta * Mb);
  if (choleskySolver.info() != Eigen::Success) {
    throw std::runtime_error("Decomposition failed");
  }
  FMCA::Vector u = choleskySolver.solve(F_comp + beta * G_comp);
  FMCA::Vector numerical_solution = hst_sources.inverseSampletTransform(u);
//  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
// solver.compute(stiffness + beta * Mb);
// FMCA::Vector u = solver.solve(F_comp + beta * G_comp);

  //std::cout << numerical_solution << std::endl;
  FMCA::Vector numerical_sol = hst_sources.inverseSampletTransform(u);
  FMCA::Vector numerical_sol_reordered = numerical_sol(permutationVector(hst_sources));
  std::cout << numerical_sol << std::endl;
  // FMCA::Vector analytical_sol = P_sources.row(0).array().pow(3) + 1.0;
  // FMCA::Scalar err = ((numerical_sol - analytical_sol).squaredNorm()) /
  //                    (analytical_sol.squaredNorm());
  // std::cout << "error =         " << err << std::endl;
  return 0;
}
