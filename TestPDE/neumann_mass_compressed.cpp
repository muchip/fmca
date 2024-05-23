#include <cstdlib>
#include <iostream>
//
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
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "Uzawa.h"
#include "read_files_txt.h"

#define DIM 2

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
  int N_WEIGHTS;
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  FMCA::Matrix P_quad_border;
  FMCA::Vector w_vec;
  //////////////////////////////////////////////////////////////////////////////
  readTXT("data/vertices_circle.txt", P_sources, NPTS_SOURCE, 2);
  readTXT("data/baricenters_circle.txt", P_quad, NPTS_QUAD, 2);
  readTXT("data/triangles_volumes_circle.txt", w_vec, N_WEIGHTS);
  //////////////////////////////////////////////////////////////////////////////
  // Right hand side f = 0
  FMCA::Vector f(NPTS_QUAD);
  for (int i = 0; i < NPTS_QUAD; ++i) {
    f(i) = exp(-sqrt((P_quad(0, i)) * (P_quad(0, i)) +
                     (P_quad(1, i)) * (P_quad(1, i))) /
               10);
  }
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 6;
  const FMCA::Scalar threshold = 1e-6;
  const FMCA::Scalar MPOLE_DEG = 6;
  const FMCA::Scalar sigma = 0.5;
  //////////////////////////////////////////////////////////////////////////////
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const Moments mom_quad(P_quad, MPOLE_DEG);
  const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
  const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
  H2SampletTree hst_sources(mom_sources, samp_mom_sources, 0, P_sources);
  H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Number of source points = " << NPTS_SOURCE << std::endl;
  std::cout << "Number of quad points = " << NPTS_QUAD << std::endl;
  std::cout << "Number of weights = " << N_WEIGHTS << std::endl;
  std::cout << "minimum element:                     "
            << *std::min_element(w_vec.begin(), w_vec.end()) << std::endl;
  std::cout << "maximum element:                     "
            << *std::max_element(w_vec.begin(), w_vec.end()) << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "sigma = " << sigma << std::endl;
  std::cout << "eta = " << eta << std::endl;
  std::cout << "dtilde = " << dtilde << std::endl;
  std::cout << "threshold = " << threshold << std::endl;
  std::cout << "MPOLE_DEG = " << MPOLE_DEG << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  FMCA::Vector f_reordered = f(permutationVector(hst_quad));
  // Weights matrix
  FMCA::Vector w_perm = w_vec(permutationVector(hst_quad));
  FMCA::SparseMatrix<FMCA::Scalar> W(w_perm.size(), w_perm.size());
  for (int i = 0; i < w_perm.size(); ++i) {
    W.insert(i, i) = w_perm(i);
  }
  FMCA::SparseMatrixEvaluator mat_eval_weights(W);
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Wcomp;
  Wcomp.init(hst_quad, eta, threshold);
  T.tic();
  Wcomp.compress(mat_eval_weights);
  const auto &trips_weights = Wcomp.triplets();
  Eigen::SparseMatrix<double> Wcomp_Sparse(NPTS_QUAD, NPTS_QUAD);
  Wcomp_Sparse.setFromTriplets(trips_weights.begin(), trips_weights.end());
  Wcomp_Sparse.makeCompressed();
  //////////////////////////////////////////////////////////////////////////////
  const FMCA::CovarianceKernel kernel_funtion("MATERN32", sigma);
  const usMatrixEvaluatorKernel mat_eval_kernel(mom_sources, mom_quad,
                                                kernel_funtion);
  // Kernel compression
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Kcomp;
  Kcomp.init(hst_sources, hst_quad, eta, threshold);
  T.tic();
  Kcomp.compress(mat_eval_kernel);
  const auto &trips = Kcomp.triplets();
  Eigen::SparseMatrix<double> Kcomp_Sparse(NPTS_SOURCE, NPTS_QUAD);
  Kcomp_Sparse.setFromTriplets(trips.begin(), trips.end());
  Kcomp_Sparse.makeCompressed();
  Eigen::SparseMatrix<double> mass =
      Kcomp_Sparse *
      (Wcomp_Sparse.selfadjointView<Eigen::Upper>() * Kcomp_Sparse.transpose())
          .eval();
  //////////////////////////////////////////////////////////////////////////////
  // F_comp = right hand side 1
  FMCA::Vector rhs =
      Kcomp_Sparse * (Wcomp_Sparse.selfadjointView<Eigen::Upper>() *
                      Kcomp_Sparse.transpose() * hst_quad.sampletTransform(f))
                         .eval();
  std::cout << "F_comp created" << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // // solve
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::MetisOrdering<int>> solver;
  solver.compute(mass);
  FMCA::Vector u = solver.solve(rhs);
  FMCA::Vector numerical_sol = hst_sources.inverseSampletTransform(u);
  
  FMCA::Vector KU = kernel_funtion.eval(P_sources, P_sources) * numerical_sol;
  FMCA::Vector analytical_sol(NPTS_SOURCE);
  for (int i = 0; i < NPTS_SOURCE; ++i) {
    analytical_sol(i) = exp(-sqrt((P_sources(0, i)) * (P_sources(0, i)) +
                                  (P_sources(1, i)) * (P_sources(1, i))) /
                            10);
  }
  std::cout << "residual error:    " << ((mass)*numerical_sol - rhs).norm()
            << std::endl;
  std::cout << "error:    "
            << (KU - analytical_sol).squaredNorm() /
                   analytical_sol.squaredNorm()
            << std::endl;
  FMCA::IO::print2m("solution_mass.m", "sol_mass", KU, "w");
  FMCA::IO::print2m("error_mass_circle.m", "error_mass_circle",
                    KU - analytical_sol, "w");
  return 0;
}
