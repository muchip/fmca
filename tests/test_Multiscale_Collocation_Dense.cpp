
#include <iostream>
#include <random>
////////////
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/MetisSupport>
#include "../FMCA/CovarianceKernel"
#include "../FMCA/KernelInterpolation"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluatorDerivatives =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;


template <typename MatrixType>
FMCA::Scalar power_iteration_largest_eig(const MatrixType& A, int maxit = 100, FMCA::Scalar tol = 1e-10) {
  using Scalar = typename MatrixType::Scalar;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Random(A.rows());
  x.normalize();
  Scalar lambda_old = Scalar(0);

  for (int k = 0; k < maxit; ++k) {
    auto y = A * x;
    Scalar nrm = y.norm();
    if (nrm == Scalar(0)) return Scalar(0);
    x = y / nrm;
    Scalar lambda = x.dot(A * x);
    if (std::abs(lambda - lambda_old) <= tol * (Scalar(1) + std::abs(lambda))) return std::abs(lambda);
    lambda_old = lambda;
  }
  return std::abs(lambda_old);
}

template <typename MatrixType>
FMCA::Scalar inverse_power_iteration_smallest_eig(const MatrixType& A, int maxit = 100, FMCA::Scalar tol = 1e-10) {
  using Scalar = typename MatrixType::Scalar;
  Eigen::LDLT<MatrixType> ldlt(A);
  if (ldlt.info() != Eigen::Success) return std::numeric_limits<FMCA::Scalar>::quiet_NaN();

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Random(A.rows());
  x.normalize();
  Scalar lambda_old = Scalar(0);

  for (int k = 0; k < maxit; ++k) {
    auto y = ldlt.solve(x);
    Scalar nrm = y.norm();
    if (nrm == Scalar(0)) return std::numeric_limits<FMCA::Scalar>::quiet_NaN();
    x = y / nrm;
    Scalar lambda = x.dot(A * x);
    if (std::abs(lambda - lambda_old) <= tol * (Scalar(1) + std::abs(lambda))) return std::abs(lambda);
    lambda_old = lambda;
  }
  return std::abs(lambda_old);
}

template <typename MatrixType>
FMCA::Scalar fast_condition_estimate(const MatrixType& A, int maxit = 100, FMCA::Scalar tol = 1e-10) {
  FMCA::Scalar lmax = power_iteration_largest_eig(A, maxit, tol);
  FMCA::Scalar lmin = inverse_power_iteration_smallest_eig(A, maxit, tol);
  if (!std::isfinite(lmax) || !std::isfinite(lmin) || lmin == 0.0) {
    return std::numeric_limits<FMCA::Scalar>::quiet_NaN();
  }
  return lmax / lmin;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main() {
  FMCA::Tictoc T;
  int NPTS_INTERIOR_1 = 256;  
  int NPTS_BORDER_1   = 16;   

  int NPTS_INTERIOR_2 = 1024;  
  int NPTS_BORDER_2   = 32;   

  int NPTS_INTERIOR_3 = 4096;  
  int NPTS_BORDER_3   = 64;  

  int NPTS_EVAL     = 50000;

  ////////////////////////// P1 P2 P3 //////////////////////////////////
  // Interior: uniform grid on (0,1)^2, excluding boundary
  FMCA::Matrix P_interior_1(2, NPTS_INTERIOR_1);
  {
    int n = (int)std::floor(std::sqrt(NPTS_INTERIOR_1));
    FMCA::Scalar h = 1.0 / (n + 1);
    int idx = 0;
    for (int ix = 1; ix <= n; ++ix)
      for (int iy = 1; iy <= n; ++iy) {
        P_interior_1(0, idx) = ix * h;
        P_interior_1(1, idx) = iy * h;
        ++idx;
      }
  }

  FMCA::Matrix P_interior_2(2, NPTS_INTERIOR_2);
  {
    int n = (int)std::floor(std::sqrt(NPTS_INTERIOR_2));
    FMCA::Scalar h = 1.0 / (n + 1);
    int idx = 0;
    for (int ix = 1; ix <= n; ++ix)
      for (int iy = 1; iy <= n; ++iy) {
        P_interior_2(0, idx) = ix * h;
        P_interior_2(1, idx) = iy * h;
        ++idx;
      }
  }

  FMCA::Matrix P_interior_3(2, NPTS_INTERIOR_3);
  {
    int n = (int)std::floor(std::sqrt(NPTS_INTERIOR_3));
    FMCA::Scalar h = 1.0 / (n + 1);
    int idx = 0;
    for (int ix = 1; ix <= n; ++ix)
      for (int iy = 1; iy <= n; ++iy) {
        P_interior_3(0, idx) = ix * h;
        P_interior_3(1, idx) = iy * h;
        ++idx;
      }
  }
  
  // Border: uniform points on the 4 sides of [0,1]^2, corners excluded
  FMCA::Matrix P_border_1(2, NPTS_BORDER_1);
  {
    int per_side = NPTS_BORDER_1 / 4;
    FMCA::Scalar h = 1.0 / (per_side + 1);
    int col = 0;
    for (int i = 1; i <= per_side; ++i) { P_border_1(0,col)=i*h; P_border_1(1,col)=0.0; ++col; }
    for (int i = 1; i <= per_side; ++i) { P_border_1(0,col)=i*h; P_border_1(1,col)=1.0; ++col; }
    for (int i = 1; i <= per_side; ++i) { P_border_1(0,col)=0.0; P_border_1(1,col)=i*h; ++col; }
    for (int i = 1; i <= per_side; ++i) { P_border_1(0,col)=1.0; P_border_1(1,col)=i*h; ++col; }
  }

  FMCA::Matrix P_border_2(2, NPTS_BORDER_2);
  {
    int per_side = NPTS_BORDER_2 / 4;
    FMCA::Scalar h = 1.0 / (per_side + 1);
    int col = 0;
    for (int i = 1; i <= per_side; ++i) { P_border_2(0,col)=i*h; P_border_2(1,col)=0.0; ++col; }
    for (int i = 1; i <= per_side; ++i) { P_border_2(0,col)=i*h; P_border_2(1,col)=1.0; ++col; }
    for (int i = 1; i <= per_side; ++i) { P_border_2(0,col)=0.0; P_border_2(1,col)=i*h; ++col; }
    for (int i = 1; i <= per_side; ++i) { P_border_2(0,col)=1.0; P_border_2(1,col)=i*h; ++col; }
  }

  FMCA::Matrix P_border_3(2, NPTS_BORDER_3);
  {
    int per_side = NPTS_BORDER_3 / 4;
    FMCA::Scalar h = 1.0 / (per_side + 1);
    int col = 0;
    for (int i = 1; i <= per_side; ++i) { P_border_3(0,col)=i*h; P_border_3(1,col)=0.0; ++col; }
    for (int i = 1; i <= per_side; ++i) { P_border_3(0,col)=i*h; P_border_3(1,col)=1.0; ++col; }
    for (int i = 1; i <= per_side; ++i) { P_border_3(0,col)=0.0; P_border_3(1,col)=i*h; ++col; }
    for (int i = 1; i <= per_side; ++i) { P_border_3(0,col)=1.0; P_border_3(1,col)=i*h; ++col; }
  }

  int N_1 = NPTS_INTERIOR_1 + NPTS_BORDER_1;
  int N_2 = NPTS_INTERIOR_2 + NPTS_BORDER_2;
  int N_3 = NPTS_INTERIOR_3 + NPTS_BORDER_3;

  FMCA::Matrix P_1(2, N_1);
  P_1.leftCols(NPTS_INTERIOR_1)  = P_interior_1;
  P_1.rightCols(NPTS_BORDER_1)   = P_border_1;

  FMCA::Matrix P_2(2, N_2);
  P_2.leftCols(NPTS_INTERIOR_2)  = P_interior_2;
  P_2.rightCols(NPTS_BORDER_2)   = P_border_2;

  FMCA::Matrix P_3(2, N_3);
  P_3.leftCols(NPTS_INTERIOR_3)  = P_interior_3;
  P_3.rightCols(NPTS_BORDER_3)   = P_border_3;

  FMCA::Matrix P_eval(2, NPTS_EVAL);
  P_eval = (FMCA::Matrix::Random(2, NPTS_EVAL).array() + 1.0) / 2.0;

  ///////////////////////////////////// K11 /////////////////////////////////////////
  const FMCA::Index dtilde = 5;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const FMCA::Scalar nu = 10.0;

  const Moments mom_1(P_1, mpole_deg);
  const SampletMoments samp_mom_1(P_1, dtilde - 1);
  const H2SampletTree hst_1(mom_1, samp_mom_1, 0, P_1);
  const FMCA::Vector minvecP_1 = minDistanceVector(hst_1, P_1);
  FMCA::Scalar fill_distanceP_1 = minvecP_1.maxCoeff();
  std::cout << "---------------------------------------" << std::endl;
  std::cout << "fill_distanceP_1: " << fill_distanceP_1 << std::endl;

  // FMCA::Scalar h1 = 1.0 / (std::sqrt(NPTS_INTERIOR_1) + 1);

  FMCA::Scalar l_1 = nu * fill_distanceP_1;
  FMCA::CovarianceKernel function("MATERN52", l_1);
  FMCA::Matrix Kb_1 = function.eval(P_border_1, P_1);

  FMCA::GradKernel function_der2X("MATERN52_SECOND_DERIVATIVE",l_1, 1.0, 0);
  FMCA::GradKernel function_der2Y("MATERN52_SECOND_DERIVATIVE",l_1, 1.0, 1);
  FMCA::Matrix KI0_1 = function_der2X.eval(P_interior_1, P_1);
  FMCA::Matrix KI1_1 = function_der2Y.eval(P_interior_1, P_1);

  FMCA::Matrix KI_1 = KI0_1 + KI1_1;
  FMCA::Matrix K_11(N_1,N_1);
  K_11 << KI_1, Kb_1;
  FMCA::Scalar cond_1 = fast_condition_estimate(K_11);
  std::cout << "cond_1: " << cond_1 << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // RHS and BC
  FMCA::Vector u_bc_1(NPTS_BORDER_1);
  u_bc_1.setZero();

  // -Delta(sin(pi*x)*sin(pi*y)) = 2*pi^2*sin(pi*x)*sin(pi*y)
  FMCA::Vector f_1(NPTS_INTERIOR_1);
  for (int i = 0; i < NPTS_INTERIOR_1; ++i) {
    FMCA::Scalar x = P_interior_1(0, i), y = P_interior_1(1, i);
    f_1[i] = - 2 * FMCA_PI * FMCA_PI * sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }

  FMCA::Vector f_rhs_1(NPTS_BORDER_1 + NPTS_INTERIOR_1);
  f_rhs_1 << f_1, u_bc_1;

  // risolvere sistema lineare 
  T.tic();
  FMCA::Vector alpha1 = K_11.colPivHouseholderQr().solve(f_rhs_1);
  T.toc("QR solve level 1");
  FMCA::Vector r1 = K_11 * alpha1 - f_rhs_1;
  std::cout << "Relative res K11: " << r1.norm() / f_rhs_1.norm() << std::endl;

  /////////////////////////////////// K22 ///////////////////////////////////////////
  const Moments mom_2(P_2, mpole_deg);
  const SampletMoments samp_mom_2(P_2, dtilde - 1);
  const H2SampletTree hst_2(mom_2, samp_mom_2, 0, P_2);
  const FMCA::Vector minvecP_2 = minDistanceVector(hst_2, P_2);
  FMCA::Scalar fill_distanceP_2 = minvecP_2.maxCoeff();
  std::cout << "---------------------------------------" << std::endl;
  std::cout << "fill_distanceP_2: " << fill_distanceP_2 << std::endl;


  FMCA::Scalar l_2 = nu * fill_distanceP_2;
  FMCA::CovarianceKernel function_2("MATERN52", l_2);
  FMCA::Matrix Kb_2 = function_2.eval(P_border_2, P_2);

  FMCA::GradKernel function_der2X_2("MATERN52_SECOND_DERIVATIVE",l_2, 1.0, 0);
  FMCA::GradKernel function_der2Y_2("MATERN52_SECOND_DERIVATIVE",l_2, 1.0, 1);
  FMCA::Matrix KI0_2 = function_der2X_2.eval(P_interior_2, P_2);
  FMCA::Matrix KI1_2 = function_der2Y_2.eval(P_interior_2, P_2);

  FMCA::Matrix KI_2 = KI0_2 + KI1_2;
  FMCA::Matrix K_22(N_2,N_2);
  K_22 << KI_2, Kb_2;
  FMCA::Scalar cond_2 = fast_condition_estimate(K_22);
  std::cout << "cond_2: " << cond_2 << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // RHS and BC
  FMCA::Vector u_bc_2(NPTS_BORDER_2);
  u_bc_2.setZero();

  // -Delta(sin(pi*x)*sin(pi*y)) = 2*pi^2*sin(pi*x)*sin(pi*y)
  FMCA::Vector f_2(NPTS_INTERIOR_2);
  for (int i = 0; i < NPTS_INTERIOR_2; ++i) {
    FMCA::Scalar x = P_interior_2(0, i), y = P_interior_2(1, i);
    f_2[i] = - 2 * FMCA_PI * FMCA_PI * sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }

  FMCA::Vector f_rhs_2(NPTS_BORDER_2 + NPTS_INTERIOR_2);
  f_rhs_2 << f_2, u_bc_2;

  //////////////////////////////////// K21 //////////////////////////////////////////
  
  FMCA::Matrix Kb_21 = function.eval(P_border_2, P_1);

  FMCA::Matrix KI0_21 = function_der2X.eval(P_interior_2, P_1);
  FMCA::Matrix KI1_21 = function_der2Y.eval(P_interior_2, P_1);

  FMCA::Matrix KI_21 = KI0_21 + KI1_21;

  FMCA::Matrix K_21(N_2, N_1);
  K_21 << KI_21, Kb_21;

  f_rhs_2 = f_rhs_2 - K_21*alpha1;

  //////////////////////////////////////////////////////////////////////////////
  // risolvere sistema lineare 
  T.tic();
  FMCA::Vector alpha2 = K_22.colPivHouseholderQr().solve(f_rhs_2);
  T.toc("QR solve level 2");
  FMCA::Vector r2 = K_22 * alpha2 - f_rhs_2;
  std::cout << "Relative res K22: " << r2.norm() / f_rhs_2.norm() << std::endl;

  ////////////////////////////////////// K33 ////////////////////////////////////////
  const Moments mom_3(P_3, mpole_deg);
  const SampletMoments samp_mom_3(P_3, dtilde - 1);
  const H2SampletTree hst_3(mom_3, samp_mom_3, 0, P_3);
  const FMCA::Vector minvecP_3 = minDistanceVector(hst_3, P_3);
  FMCA::Scalar fill_distanceP_3 = minvecP_3.maxCoeff();
  std::cout << "---------------------------------------" << std::endl;
  std::cout << "fill_distanceP_3: " << fill_distanceP_3 << std::endl;


  FMCA::Scalar l_3 = nu * fill_distanceP_3;
  FMCA::CovarianceKernel function_3("MATERN52", l_3);
  FMCA::Matrix Kb_3 = function_3.eval(P_border_3, P_3);

  FMCA::GradKernel function_der2X_3("MATERN52_SECOND_DERIVATIVE",l_3, 1.0, 0);
  FMCA::GradKernel function_der2Y_3("MATERN52_SECOND_DERIVATIVE",l_3, 1.0, 1);
  FMCA::Matrix KI0_3 = function_der2X_3.eval(P_interior_3, P_3);
  FMCA::Matrix KI1_3 = function_der2Y_3.eval(P_interior_3, P_3);

  FMCA::Matrix KI_3 = KI0_3 + KI1_3;
  FMCA::Matrix K_33(N_3,N_3);
  K_33 << KI_3, Kb_3;
  FMCA::Scalar cond_3 = fast_condition_estimate(K_33);
  std::cout << "cond_3: " << cond_3 << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // RHS and BC
  FMCA::Vector u_bc_3(NPTS_BORDER_3);
  u_bc_3.setZero();

  // -Delta(sin(pi*x)*sin(pi*y)) = 2*pi^2*sin(pi*x)*sin(pi*y)
  FMCA::Vector f_3(NPTS_INTERIOR_3);
  for (int i = 0; i < NPTS_INTERIOR_3; ++i) {
    FMCA::Scalar x = P_interior_3(0, i), y = P_interior_3(1, i);
    f_3[i] = - 2 * FMCA_PI * FMCA_PI * sin(FMCA_PI * x) * sin(FMCA_PI * y);
  }

  /////////////////////////////////// K31 ///////////////////////////////////////////
  FMCA::Matrix Kb_31 = function.eval(P_border_3, P_1);
  FMCA::Matrix KI0_31 = function_der2X.eval(P_interior_3, P_1);
  FMCA::Matrix KI1_31 = function_der2Y.eval(P_interior_3, P_1);

  FMCA::Matrix KI_31 = KI0_31 + KI1_31;

  FMCA::Matrix K_31(N_3, N_1);
  K_31 << KI_31, Kb_31;

  /////////////////////////////////// K32 ///////////////////////////////////////////
  
  FMCA::Matrix Kb_32 = function_2.eval(P_border_3, P_2);

  FMCA::Matrix KI0_32 = function_der2X_2.eval(P_interior_3, P_2);
  FMCA::Matrix KI1_32 = function_der2Y_2.eval(P_interior_3, P_2);

  FMCA::Matrix KI_32 = KI0_32 + KI1_32;

  FMCA::Matrix K_32(N_3, N_2);
  K_32 << KI_32, Kb_32;
  //////////////////////////////////////////////////////////////////////////////

  // risolvere sistema lineare 

  FMCA::Vector f_rhs_3(NPTS_BORDER_3 + NPTS_INTERIOR_3);
  f_rhs_3 << f_3, u_bc_3;

  f_rhs_3 = f_rhs_3 - K_32*alpha2 - K_31*alpha1;

  T.tic();
  FMCA::Vector alpha3 = K_33.colPivHouseholderQr().solve(f_rhs_3);
  T.toc("QR solve level 3");
  FMCA::Vector r3 = K_33 * alpha3 - f_rhs_3;
  std::cout << "Relative res K33: " << r3.norm() / f_rhs_3.norm() << std::endl;
  //////////////////////////////////////////////////////////////////////////////

  // P eval 
  FMCA::Matrix K_eval_1 = function.eval(P_eval, P_1);
  FMCA::Vector u_num_1 = K_eval_1 * alpha1;

  FMCA::Matrix K_eval_2 = function_2.eval(P_eval, P_2);
  FMCA::Vector u_num_2 = K_eval_2 * alpha2;

  FMCA::Matrix K_eval_3 = function_3.eval(P_eval, P_3);
  FMCA::Vector u_num_3 = K_eval_3 * alpha3;

  FMCA::Vector u_num = u_num_1 + u_num_2 + u_num_3;

  FMCA::Vector u_exact(NPTS_EVAL);
  for (int i = 0; i < NPTS_EVAL; ++i)
    u_exact[i] = sin(FMCA_PI * P_eval(0,i)) * sin(FMCA_PI * P_eval(1,i));
  
  std::cout << "---------------------------------------" << std::endl;
  std::cout << "L2 relative error : " << (u_num - u_exact).norm() / u_exact.norm() << std::endl;
  
}