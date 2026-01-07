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
inline Scalar Phi(const MatrixReplacement &A, const Vector &b, const Vector &w,
                  const Vector &x) {
  return 0.5 * (b - A * x).squaredNorm() + x.cwiseAbs().dot(w);
}

/**
 *  \brief Soft-Shrinkage operator
 **/
inline Vector SS(const Vector &x, const Vector &w) {
  return (x.array().abs() - w.array()).max(0.0) * x.array().sign();
}

/**
 *  \brief normal map
 **/
template <typename MatrixReplacement>
inline Vector Fnormal(const MatrixReplacement &A, const Vector &b,
                      const Vector &w, const Vector &x, const Scalar lambda) {
  const Vector SSx = SS(x, lambda * w);
  return A.transpose() * (A * SSx - b).eval() + (1. / lambda) * (x - SSx);
}

template <typename MatrixReplacement>
inline Scalar Htau(const MatrixReplacement &A, const Vector &b, const Vector &w,
                   const Vector &x, const Scalar lambda, const Scalar tau) {
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
inline Vector activeSet(const Vector &x, const Vector &w) {
  return (x.array().abs() > w.array()).cast<Scalar>();
}

template <typename SparseMatrix>
Vector SSN(const SparseMatrix &A, const Vector &b, const Vector &w,
           const Vector &x0, ActiveSetManager &asmgr, Index steps = 1000,
           Scalar tol = 1e-6) {
  const Index npts = A.rows();
  std::vector<Index> aidcs, iidcs;
  Vector x = x0, g = x0, u = x0, r = x0, active, inactive;
  Scalar cond = 0, gamma = 1., phi = 0, phimin = Phi(A, b, w, x);
  Index n_active = 0, iter = 0, n_gamma = 0;
  ;
  //////////////////////////////////////////////////////////////////////////
  for (; iter < steps; ++iter) {
    g = A * (A * x - b);
    u = x - gamma * g;
    r = x - SS(u, gamma * w);
    active = activeSet(u, gamma * w);
    inactive = Vector::Ones(npts) - active;
    n_active = active.sum();
    std::cout << "\rres: " << r.cwiseAbs().maxCoeff() << " active: " << n_active
              << std::flush;
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

template <typename SparseMatrix>
Vector TRSSN(const SparseMatrix &A, const Vector &b, const Vector &w,
             const Vector &x0, ActiveSetManager &asmgr,
             const Scalar lambda = 0.1, const Scalar eta = 0.5,
             const Scalar tau = 0.2, const Index steps = 1000,
             const Scalar tol = 1e-6) {
  const Scalar npts = A.rows();
  const Scalar nu = 0.5 * tau;
  const Scalar delta_min = 1e-6;
  const Scalar delta_max = 10.;
  std::vector<Index> aidcs, iidcs;
  Vector x = x0, s = x0, fnor, active, inactive;
  Scalar delta = 1, ared = 0, pred = 0, rho = 0, cond = 0;
  const Scalar eta2 = eta + 0.05 * (1 - eta);
  Index iter = 0, n_active = 0;
  fnor = Fnormal(A, b, w, x, lambda);
  Scalar norm_fnor = fnor.norm();
  //////////////////////////////////////////////////////////////////////////
  do {
    active = activeSet(x, lambda * w);
    inactive = Vector::Ones(npts) - active;
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
    // compute Newton correction
    {
      Vector arhs(n_active);
      for (Index i = 0; i < aidcs.size(); ++i) arhs(i) = -fnor(aidcs[i]);
      const Matrix VSinv = asmgr.activeVSinv(aidcs);
      cond = asmgr.sactive()(0) / asmgr.sactive()(asmgr.sactive().size() - 1);
      cond *= cond;
      if (cond > 1e15) std::cout << "ill conditioned" << std::endl;
      const Vector ax = VSinv * (VSinv.transpose() * arhs).eval();
      // set active components to compute inactive part
      s.setZero();
      for (Index i = 0; i < aidcs.size(); ++i) s(aidcs[i]) = ax(i);
      s = -lambda * (fnor + A.transpose() * (A * s).eval());
      // reset active part
      for (Index i = 0; i < aidcs.size(); ++i) s(aidcs[i]) = ax(i);
      // scale s by step size
      s *= std::min({1., Scalar(delta / s.norm())});
    }
    ared = Htau(A, b, w, x, lambda, tau) - Htau(A, b, w, x + s, lambda, tau);
    const Scalar scal = std::min({lambda, delta, lambda * norm_fnor});
    pred = 0.5 * tau * norm_fnor * scal +
           nu * norm_fnor / scal *
               (SS(x + s, lambda * w) - SS(x, lambda * w)).squaredNorm();
    if (pred <= 0) {
      rho = 0;
    } else {
      rho = ared / pred;
    }
    if (rho >= eta) x += s;
    if (rho < eta)
      delta *= 0.5;
    else if (rho > eta2)
      delta *= 2;
    delta = std::max(delta_min, std::min(delta, delta_max));
    ++iter;
    fnor = Fnormal(A, b, w, x, lambda);
    norm_fnor = fnor.norm();
    std::cout << "delta: " << delta << " nactive: " << n_active
              << " res: " << norm_fnor << std::endl;
  } while (iter < steps && norm_fnor > tol);
  std::cout << std::endl
            << "dict size: " << asmgr.matrixU().cols()
            << " active size: " << aidcs.size() << " iterations: " << iter
            << std::endl;
  return x;
}

}  // namespace FMCA
#endif
