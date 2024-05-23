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
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "../FMCA/src/util/IO.h"
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
  w_vec = w_vec * 1 / 4;
  FMCA::Vector w_vec_border(NPTS_QUAD_BORDER);
  w_vec_border.setOnes();
  w_vec_border = w_vec_border * 1 / 2;
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
  //////////////////////////////////////////////////////////////////////////////
  // Right hand side f = 0
  FMCA::Vector f;
  f.resize(NPTS_QUAD);
  for (int i = 0; i < NPTS_QUAD; ++i) {
    f(i) = (exp(-sqrt(P_quad(0,i)*P_quad(0,i)-10*P_quad(0,i)+P_quad(1,i)*P_quad(1,i)-4*P_quad(1,i)+29)))*
    (2-1/(sqrt(P_quad(0,i)*P_quad(0,i)-10*P_quad(0,i)+P_quad(1,i)*P_quad(1,i)-4*P_quad(1,i)+29)));
  }
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 6;
  const FMCA::Scalar threshold = 1e-6;
  const FMCA::Scalar MPOLE_DEG = 6;
  const FMCA::Scalar sigma = 1/sqrt(2);
  const FMCA::Scalar beta = 500;
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
  // Weights matrix
  Eigen::MatrixXd W(w_vec.size(), w_vec.size());
  for (int i = 0; i < w_vec.size(); ++i) {
    W(i, i) = w_vec(i);
  }
  //////////////////////////////////////////////////////////////////////////////
  // Weights matrix border
  Eigen::MatrixXd W_border(w_vec_border.size(), w_vec_border.size());
  for (int i = 0; i < w_vec_border.size(); ++i) {
    W_border(i, i) = w_vec_border(i);
  }
  //////////////////////////////////////////////////////////////////////////////
  const FMCA::CovarianceKernel kernel_funtion("MATERN32", sigma);
  Eigen::MatrixXd Kernel = kernel_funtion.eval(P_sources, P_quad);
  Eigen::MatrixXd mass = Kernel*W*Kernel.transpose();
  //////////////////////////////////////////////////////////////////////////////
  Eigen::MatrixXd stiffness(NPTS_SOURCE, NPTS_SOURCE);
  stiffness.setZero();

  for (int i = 0; i < DIM; ++i) {
    std::cout << std::string(80, '-') << std::endl;
    const FMCA::GradKernel function("MATERN32", sigma, 1, i);
    Eigen::MatrixXd gradKernel = function.eval(P_sources, P_quad);
    //////////////////////////////////////////////////////////////////////////////
    // eigen multiplication
    T.tic();
    Eigen::MatrixXd gradk = gradKernel * (W * gradKernel.transpose()).eval();
    stiffness += gradk;
  }
  //////////////////////////////////////////////////////////////////////////////
  std::cout << std::string(80, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // F_comp = right hand side 1
  FMCA::Vector rhs =
      Kernel * (W * f).eval();
  std::cout << "F_comp created" << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  std::cout << "Frobenius norm stiffness: " << stiffness.norm() << std::endl;
  std::cout << "determinant: "
            << Eigen::MatrixXd(stiffness).determinant()
            << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // // solve
  Eigen::HouseholderQR<Eigen::MatrixXd> solver;
  solver.compute(2*stiffness + mass);
  FMCA::Vector numerical_sol = solver.solve(rhs);
  std::cout << "numerical sol done"<< std::endl;
  FMCA::Vector KU = kernel_funtion.eval(P_sources, P_sources) * numerical_sol;
  FMCA::Vector analytical_sol(NPTS_SOURCE);
  for (int i = 0; i<NPTS_SOURCE; ++i){
    analytical_sol(i) = exp(-sqrt((P_sources(0,i)-5)*(P_sources(0,i)-5) + (P_sources(1,i)-2)*(P_sources(1,i)-2)));
  }
  std::cout << (KU-analytical_sol).norm()/analytical_sol.norm() << std::endl;
  FMCA::IO::print2m("solution.m", "sol", KU, "w");
  return 0;
}
