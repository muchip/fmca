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
#ifndef FMCA_SSN_H_
#define FMCA_SSN_H_

#include "ActiveSetManager.h"

namespace FMCA {
/*
 *  \brief goal function ||b-Ax||_2^2+ w^T|x|
 *
 **/
template <typename MatrixReplacement>
inline Scalar Phi(const MatrixReplacement& A, const Vector& b, const Vector& w,
                  const Vector& x) {
  return 0.5 * (b - A * x).squaredNorm() + x.cwiseAbs().dot(w);
}

/**
 *  \brief Soft-Shrinkage operator
 **/
inline Vector SS(const Vector& x, const Vector& w) {
  return (x.array().abs() - w.array()).max(0.0) * x.array().sign();
}

/**
 *  \brief normal map
 **/
template <typename MatrixReplacement>
inline Vector Fnormal(const MatrixReplacement& A, const Vector& b,
                      const Vector& w, const Vector& x, const Scalar lambda) {
  const Vector SSx = SS(x, lambda * w);
  return A.transpose() * (A * SSx - b).eval() + (1. / lambda) * (x - SSx);
}

/**
 *  \brief Htau function for SSN
 *
 *  The Htau function is used in the SSN algorithm to compute the
 *  merit function at a given iterate.
 *
 *  \param A The matrix A in the problem
 *  \param b The vector b in the problem
 *  \param w The vector w in the problem
 *  \param x The iterate at which to compute the merit function
 *  \param lambda The step size parameter
 *  \param tau The penalty parameter
 *
 *  \return The value of the merit function at x
 */
template <typename MatrixReplacement>
inline Scalar Htau(const MatrixReplacement& A, const Vector& b, const Vector& w,
                   const Vector& x, const Scalar lambda, const Scalar tau) {
  const Vector SSx = SS(x, lambda * w);
  const Vector aux = (A * SSx - b).eval();
  const Scalar psi = 0.5 * aux.squaredNorm() + SSx.cwiseAbs().dot(w);
  const Scalar fnor2 =
      (A.transpose() * aux + (1. / lambda) * (x - SSx)).squaredNorm();
  return psi + 0.5 * tau * lambda * fnor2;
}

/**
 *  \brief active set
 **/
inline Vector activeSet(const Vector& x, const Vector& w) {
  return (x.array().abs() > w.array()).cast<Scalar>();
}

/**
 *  \brief SemiSmooth Newton (SSN) algorithm
 *
 *  This function implements the SSN algorithm for solving the following
 *  problem:
 *
 *  min ||b-Ax||_2^2+ w^T|x|
 *
 *  where A is a sparse matrix, b and w are vectors, and x is the
 *  variable to be optimized.
 *
 *  The algorithm is an iterative method that starts from an initial
 *  guess and converges to the optimal solution. At each iteration,
 *  the algorithm updates the active set of variables and computes the
 *  Newton correction. The Newton correction is then used to update the
 *  variables.
 *
 *  \param A The sparse matrix A in the problem
 *  \param b The vector b in the problem
 *  \param w The vector w in the problem
 *  \param x0 The initial guess for the variable x
 *  \param asmgr The ActiveSetManager that is used to store the active
 *  set of variables
 *  \param steps The maximum number of iterations
 *  \param tol The tolerance for convergence
 *
 *  \return The optimal solution x
 */
template <typename SparseMatrix>
Vector SSN(const SparseMatrix& A, const Vector& b, const Vector& w,
           const Vector& x0, ActiveSetManager& asmgr, Index steps = 1000,
           Scalar tol = 1e-6) {
  const Index npts = A.rows();
  std::vector<Index> aidcs, iidcs;
  Vector x = x0, g = x0, u = x0, r = x0, active, inactive;
  Scalar cond = 0, gamma = 1., phi = 0, phimin = Phi(A, b, w, x);
  Index n_active = 0, iter = 0, n_gamma = 0;
  //////////////////////////////////////////////////////////////////////////
  for (; iter < steps; ++iter) {
    g = A * (A * x - b);
    u = x - gamma * g;
    r = x - SS(u, gamma * w);
    active = activeSet(u, gamma * w);
    inactive = Vector::Ones(npts) - active;
    n_active = active.sum();
    std::cout << "\rres: " << r.cwiseAbs().maxCoeff() << " active: " << n_active
              << std::endl;
    if (r.cwiseAbs().maxCoeff() < tol) break;
    Vector rhs = gamma * A * (A * (inactive.asDiagonal() * r).eval()).eval();
    rhs = active.asDiagonal() * (rhs - r);
    aidcs.clear();
    iidcs.clear();
    for (Index i = 0; i < active.size(); ++i)
      if (active(i))
        aidcs.push_back(i);
      else
        iidcs.push_back(i);
    if (!asmgr.indices().size())
      asmgr.init(A, aidcs);
    else
      asmgr.update(A, aidcs);

    Vector arhs(n_active);
    for (Index j = 0; j < aidcs.size(); ++j) arhs(j) = rhs(aidcs[j]);
    const Matrix VSinv = asmgr.activeVSinv(aidcs);
    cond = asmgr.sactive()(0) / asmgr.sactive()(asmgr.sactive().size() - 1);
    cond *= cond;
    if (cond > 1e15) std::cout << "ill conditioned" << std::endl;
    const Vector ax = VSinv * (VSinv.transpose() * arhs).eval() / gamma;
    gamma = 1. / asmgr.sactive()(asmgr.sactive().size() - 1);
    gamma *= gamma;
    for (Index i = 0; i < aidcs.size(); ++i) x(aidcs[i]) += ax(i);
    for (Index i = 0; i < iidcs.size(); ++i) x(iidcs[i]) = 0;
    phi = Phi(A, b, w, x);
  }
  std::cout << std::endl
            << "dict size: " << asmgr.matrixU().cols()
            << " active size: " << aidcs.size() << " iterations: " << iter
            << std::endl;
  return x;
}

/**
 *  \brief Trust Region SemiSmooth Newton (TRSSN) algorithm
 *
 *  This function implements the TRSSN algorithm for solving the following
 *  problem:
 *
 *  min ||b-Ax||_2^2+ w^T|x|
 *
 *  where A is a sparse matrix, b and w are vectors, and x is the
 *  variable to be optimized.
 *
 *  The algorithm is an iterative method that starts from an initial
 *  guess and converges to the optimal solution. At each iteration,
 *  the algorithm updates the active set of variables, computes the
 *  Newton correction, and then updates the variables.
 *
 *  \param A The sparse matrix A in the problem
 *  \param b The vector b in the problem
 *  \param w The vector w in the problem
 *  \param x0 The initial guess for the variable x
 *  \param asmgr The ActiveSetManager that is used to store the active
 *  set of variables
 *  \param lambda The step size parameter
 *  \param lambda_method The method by which to update lambda
 *  \param eta1 The minimum value of rho for acceptance
 *  \param eta2 The maximum value of rho for acceptance
 *  \param tau The penalty parameter
 *  \param nu The reduction parameter
 *  \param steps The maximum number of iterations
 *  \param tol The tolerance for convergence
 *
 *  \return The optimal solution x
 */
template <typename SparseMatrix>
Vector TRSSN(const SparseMatrix& A, const Vector& b, const Vector& w,
             const Vector& x0, ActiveSetManager& asmgr, Scalar lambda = 1.,
             const Scalar eta1 = 1e-6, const Scalar eta2 = 0.75,
             const Scalar tau = 0.5, const Scalar nu = 0.5,
             const Index steps = 1000, const Scalar tol = 1e-6) {
  const Index ndata = A.rows();  // data dimension (e.g. number of points)
  const Index npar = A.cols();   // parameter dimension (coefficients)
  const Scalar delta_min = 1e-5;
  const Scalar delta_max = 1e5;
  Scalar success_iter = 0.0;
  std::vector<Index> aidcs, iidcs;
  Vector x = x0;      // size npar
  Vector s = x0;      // size npar
  Vector fnor(npar);  // Fnormal is in parameter space
  Vector active(npar), inactive(npar);
  Scalar delta = 1.0, ared = 0.0, pred = 0.0, rho = 0.0, cond = 0.0;
  Index iter = 0, n_active = 0;
  fnor = Fnormal(A, b, w, x, lambda);  // size npar
  Scalar norm_fnor = fnor.norm();
  do {
    // active set of coefficients
    active = activeSet(x, lambda * w);  // size npar
    inactive = Vector::Ones(npar) - active;
    n_active = active.sum();

    aidcs.clear();
    iidcs.clear();
    for (Index i = 0; i < active.size(); ++i)
      if (active(i))
        aidcs.push_back(i);
      else
        iidcs.push_back(i);

    if (!asmgr.indices().size())
      asmgr.init(A, aidcs);
    else
      asmgr.update(A, aidcs);
    // Newton correction
    if (n_active) {
      Vector arhs(n_active);
      for (Index i = 0; i < aidcs.size(); ++i)
        arhs(i) = -fnor(aidcs[i]);  // fnor in param space
      const Matrix VSinv = asmgr.activeVSinv(aidcs);
      cond = asmgr.sactive()(0) / asmgr.sactive()(asmgr.sactive().size() - 1);
      cond *= cond;
      if (cond > 1e15) std::cout << "ill conditioned" << std::endl;
      const Vector ax = VSinv * (VSinv.transpose() * arhs).eval();
      s.setZero();
      for (Index i = 0; i < aidcs.size(); ++i) s(aidcs[i]) = ax(i);
      const Vector AAs = A.transpose() * (A * s).eval();  // size npar
      s = -lambda * (fnor + AAs);
      for (Index i = 0; i < aidcs.size(); ++i) s(aidcs[i]) = ax(i);
      s *= std::min(Scalar(1.), Scalar(delta / s.norm()));
    } else {
      s = -lambda * fnor;
      s *= std::min(Scalar(1.), Scalar(delta / s.norm()));
    }
    // Reduction
    ared = Htau(A, b, w, x, lambda, tau) - Htau(A, b, w, x + s, lambda, tau);
    const Scalar scal = std::min({lambda, delta, lambda * norm_fnor});
    const Scalar scal2 = std::min({delta, lambda * norm_fnor});
    Scalar nu_k = std::min(
        nu, Scalar(1e-3) *
                std::pow((success_iter * std::pow(std::log(success_iter), 2)),
                         Scalar(0.2)) *
                std::pow((SS(x + s, lambda * w) - SS(x, lambda * w)).norm(),
                         Scalar(0.2)));
    pred = 0.5 * tau * norm_fnor * scal +
           nu_k * norm_fnor / scal2 *
               (SS(x + s, lambda * w) - SS(x, lambda * w)).squaredNorm();
    // Trust-region acceptance
    rho = (pred <= 0) ? 0.0 : ared / pred;
    if (rho >= eta1) {
      x += s;
      success_iter++;
    }
    if (rho < eta1) {
      delta *= 0.5;
    } else if (rho >= eta2) {
      delta *= 2.0;
    }
    delta = std::clamp(delta, delta_min, delta_max);
    ++iter;
    fnor = Fnormal(A, b, w, x, lambda);
    norm_fnor = fnor.norm();
    std::cout << "\r" << std::string(80, ' ') << "\riter: " << iter
              << " lambda: " << lambda << " w: " << w[0] << " delta: " << delta
              << " nactive: " << n_active << " cond: " << cond
              << " res: " << norm_fnor << std::flush;
  } while (iter < steps && norm_fnor > tol);

  std::cout << "\n"
            << "dict size: " << asmgr.matrixU().cols()
            << " active size: " << aidcs.size() << " iterations: " << iter
            << std::endl;

  return SS(x, lambda * w);
}

}  // namespace FMCA
#endif
