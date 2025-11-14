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
 *  \brief active set
 **/
inline Vector activeSet(const Vector &x, const Vector &w) {
  return (x.array().abs() > w.array()).cast<Scalar>();
}

/*
 *  \brief symmetric power iteration for the largest eigenvalue
 *
 **/
template <typename SparseMatrix>
Scalar powerIteration(const SparseMatrix &A, Index steps = 20) {
  Scalar norm = 0;
  Vector x = Vector::Random(A.rows());
  for (auto i = 0; i < steps; ++i) {
    x = A * x;
    x /= x.norm();
  }
  return x.dot(A * x);
}

template <typename SparseMatrix>
Vector SSN(const SparseMatrix &A, const Vector &b, const Vector &w,
           const Vector &x0, Index steps = 1000, Scalar tol = 1e-6) {
  const Index npts = A.rows();
  std::vector<Index> aidcs, iidcs;
  Vector x = x0, g = x0, u = x0, r = x0, active, inactive;
  Scalar cond = 0, gamma = 1., phi = 0, phimin = Phi(A, b, w, x);
  Index n_active = 0, iter = 0, n_gamma = 0;
  ActiveSetManager asmgr;
  //////////////////////////////////////////////////////////////////////////
  for (; iter < steps; ++iter) {
    g = A * (A * x - b);
    u = x - gamma * g;
    r = x - SS(u, gamma * w);
    active = activeSet(u, gamma * w);
    inactive = Vector::Ones(npts) - active;
    n_active = active.sum();
    std::cout << "res: " << r.cwiseAbs().maxCoeff();
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
    std::cout << " active size: " << aidcs.size() << " " << std::flush;
    if (!asmgr.indices().size())
      asmgr.init(A, aidcs);
    else
      asmgr.update(A, aidcs);

    Vector arhs(n_active);
    for (Index j = 0; j < aidcs.size(); ++j) arhs(j) = rhs(aidcs[j]);
    const Matrix VSinv = asmgr.activeVSinv(aidcs);
    cond = asmgr.sactive()(0) / asmgr.sactive()(asmgr.sactive().size() - 1);
    cond *= cond;
    if (cond > 1e12) std::cout << "ill conditioned" << std::endl;
    const Vector ax = VSinv * (VSinv.transpose() * arhs).eval() / gamma;
    gamma = 1. / asmgr.sactive()(asmgr.sactive().size() - 1);
    gamma *= gamma;
    for (Index i = 0; i < aidcs.size(); ++i) x(aidcs[i]) += ax(i);
    for (Index i = 0; i < iidcs.size(); ++i) x(iidcs[i]) = 0;
    phi = Phi(A, b, w, x);
    std::cout << " gamma: " << gamma << " Functional: " << phi << std::endl;
  }
  std::cout << std::endl;
  return x;
}

}  // namespace FMCA
#endif
