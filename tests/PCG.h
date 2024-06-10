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
namespace FMCA {
template <typename Preconditioner, typename Matrix>
Vector pCG(const Preconditioner &E, const Matrix &A, const Vector &b,
           const Scalar ridge_parameter = 0) {
  Vector x = b;
  Vector r = b - A * x - ridge_parameter * x;
  Vector z = E.transpose() * (E * r).eval();
  Vector p = z;
  Vector Ap;
  Scalar err = r.norm() / b.norm();
  Scalar alpha = 0;
  Scalar beta = 0;
  Scalar rdotz = 0;
  Scalar rdotz_old = r.dot(z);
  Index iter = 0;
  while (err > 1e-10) {
    Ap = A * p + ridge_parameter * p;
    alpha = r.dot(z) / p.dot(Ap);
    x += alpha * p;
    r -= alpha * Ap;
    err = r.norm() / b.norm();
    z = E.transpose() * (E * r).eval();
    rdotz = r.dot(z);
    beta = rdotz / rdotz_old;
    rdotz_old = rdotz;
    p = z + beta * p;
    ++iter;
  }
  std::cout << "CG error: " << err << " iterations: " << iter << std::endl;
  return x;
}
}  // namespace FMCA
