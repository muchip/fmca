// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer, Sara Avesani
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_SEMILAGRANGIAN_KERNELINTERPOLATOR2D_H_
#define FMCA_SEMILAGRANGIAN_KERNELINTERPOLATOR2D_H_

namespace FMCA {

/**
 *  \brief Kernel-based reconstruction for the semi-Lagrangian scheme.
 *
 *  Given data sites X = {x_1, ..., x_N} and values u_i, the interpolant
 *  reads
 *      s(q) = sum_{i=1}^N c_i k(q, x_i),
 *  with the coefficients c determined by (K + ridge*I) c = u.
 *
 *    - The solve goes through `SampletKernelSolver`, which compresses K
 *      via the samplet basis and factorises a sparse symmetric system;
 *    - The evaluation goes through `MultipoleFunctionEvaluator`, which
 *      applies the H2-compressed kernel matrix at arbitrary query
 *      points.
 *
 *  Coefficients and query results are exchanged in *natural* order;
 *  the cluster reorderings happen inside FMCA.
 */

class KernelInterpolator2D {
 public:
  KernelInterpolator2D() = default;

  void init(const CovarianceKernel &kernel, const Matrix &P, Index dtilde,
            Scalar eta = 0.5, Scalar threshold = 1e-6, Scalar ridgep = 1e-8) {
    kernel_ = kernel;
    P_ = P;
    eta_ = eta;
    dtilde_ = dtilde;
    mpole_deg_ = (dtilde_ > 1) ? (2 * (dtilde_ - 1)) : 1;
    solver_.init(kernel_, P_, dtilde_, eta_, threshold, ridgep);
    solver_.compress(P_);
    solver_.factorize();
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// Recompute kernel coefficients from new function values @p u
  /// (natural order, one entry per data site).
  void compute(const Vector &u) {
    Matrix rhs = u;
    coeffs_ = solver_.solve(rhs);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// Evaluate the kernel interpolant at the query points stored as
  /// columns of @p Q (size DIM x M). A MultipoleFunctionEvaluator
  /// is constructed for each call: the eval-side cluster tree depends
  /// on @p Q which changes every RK stage, so caching it would be
  /// invalid.
  Vector evaluate(const Matrix &Q) const {
    MultipoleFunctionEvaluator mfe(kernel_, P_, Q, eta_, mpole_deg_);
    Matrix S = mfe.evaluate(P_, Q, coeffs_);
    return S.col(0);
  }

  //////////////////////////////////////////////////////////////////////////////
  Scalar compressionError() { return solver_.compressionError(P_); }

  //////////////////////////////////////////////////////////////////////////////
  Scalar interpolationResidual(const Vector &u_test) {
    compute(u_test);
    Vector s = evaluate(P_);
    return (s - u_test).norm() / u_test.norm();
  }

  //////////////////////////////////////////////////////////////////////////////
  const Matrix &points() const { return P_; }
  const CovarianceKernel &kernel() const { return kernel_; }
  const Matrix &coeffs() const { return coeffs_; }

 private:
  CovarianceKernel kernel_;
  Matrix P_;
  Scalar eta_ = 0.8;
  Index dtilde_ = 1;
  Index mpole_deg_ = 1;
  SampletKernelSolver solver_;
  Matrix coeffs_;  // N x 1, kernel coefficients in natural order
};

}  // namespace FMCA

#endif  // FMCA_SEMILAGRANGIAN_KERNELINTERPOLATOR2D_H_
