// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
// #define EIGEN_DONT_PARALLELIZE
#include <FMCA/src/util/Tictoc.h>

#include <FMCA/KernelInterpolation>

int main() {
  FMCA::Tictoc T;
  const FMCA::Index npts = 10000;
  const FMCA::Index dim = 2;
  const FMCA::CovarianceKernel kernel("EXPONENTIAL", 1.);
  const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(dim, npts).array() + 1);
  const FMCA::Scalar threshold = 1e-8;
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 5;
  const FMCA::Vector rhs = FMCA::Vector::Ones(npts);
  const FMCA::Matrix K = kernel.eval(P, P);
  FMCA::SampletKernelSolver sks;
  sks.init(kernel, P, dtilde, eta, 1e-8, 1e-6);
  sks.compress(P);
  std::cout << "fill distance:     " << sks.fill_distance() << std::endl;
  std::cout << "separation radius: " << sks.separation_radius() << std::endl;
  std::cout << "Compression error: " << sks.compressionError(P) << std::endl;
  FMCA::Scalar residual = 0;
  T.tic();
  const FMCA::Vector sol_iter = sks.solveIteratively(rhs, true, 1e-7);
  T.toc("PCG:              ");
  residual = (K * sol_iter - rhs).norm() / rhs.norm();
  std::cout << "residual           " << residual << std::endl;
  std::cout << "solver iterations: " << sks.solver_iterations() << std::endl;
  if (residual > 1e-6) return 1;
  T.tic();
  const FMCA::Vector sol_iter_np = sks.solveIteratively(rhs, false, 1e-7);
  T.toc("CG:               ");
  residual = (K * sol_iter_np - rhs).norm() / rhs.norm();
  std::cout << "residual           " << residual << std::endl;
  std::cout << "solver iterations: " << sks.solver_iterations() << std::endl;
  if (residual > 1e-6) return 1;

  T.tic();
  sks.factorize();
  const FMCA::Vector sol_direct = sks.solveDirectly(rhs);
  residual = (K * sol_direct - rhs).norm() / rhs.norm();
  std::cout << "residual           " << residual << std::endl;
  T.toc("direct solver:    ");
  if (residual > 1e-6) return 1;

  return 0;
}
