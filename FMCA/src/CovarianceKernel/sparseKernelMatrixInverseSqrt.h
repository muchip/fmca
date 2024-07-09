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
#ifndef FMCA_COVARIANCEKERNEL_SPARSEKERNELMATRIXINVERSESQRT_H_
#define FMCA_COVARIANCEKERNEL_SPARSEKERNELMATRIXINVERSESQRT_H_

namespace FMCA {

template <typename Derived>
Eigen::SparseMatrix<Scalar> sparseKernelMatrixInverseSqrt(
    const CovarianceKernel &K, const ClusterTreeBase<Derived> &CT,
    const Matrix &P, const Scalar fps = 1., const Scalar ridge_parameter = 0,
    const Scalar zeroTol = FMCA_ZERO_TOLERANCE) {
  const Vector mdv = minDistanceVector(CT, P);
  const Scalar fill_distance = mdv.maxCoeff();
  const Scalar search_radius =
      fps * fill_distance * std::abs(std::log(fill_distance));
  std::cout << std::string(28, '-') << "sparseInverseSqrt"
            << std::string(28, '-') << std::endl;
  std::cout << "fill distance:                " << fill_distance << std::endl;
  std::cout << "search radius:                " << search_radius << std::endl;
  // compute epsilon nearest neighbours for all points
  std::vector<std::vector<Index>> epsnn(P.cols());
#pragma omp parallel for
  for (Index i = 0; i < epsnn.size(); ++i) {
    epsnn[i] = epsNN(CT, P, P.col(i), search_radius);
    std::sort(epsnn[i].begin(), epsnn[i].end());
  }
  // evaluate localized inverse
  std::vector<Eigen::Triplet<Scalar>> triplets;
  // compute permutation from original order to cluster order
  std::vector<Index> inv_idcs(P.cols());
  for (FMCA::Index i = 0; i < inv_idcs.size(); ++i)
    inv_idcs[CT.indices()[i]] = i;
    // actually compute the localized inverse
#pragma omp parallel for
  for (FMCA::Index i = 0; i < epsnn.size(); ++i) {
    const FMCA::Index locN = epsnn[i].size();
    FMCA::Matrix Ploc(P.rows(), locN);
    FMCA::Index pos = 0;
    for (FMCA::Index j = 0; j < locN; ++j) {
      Ploc.col(j) = P.col(epsnn[i][j]);
      pos = epsnn[i][j] == i ? j : pos;
    }
    FMCA::Matrix Kloc = K.eval(Ploc, Ploc);
    Kloc.diagonal().array() += ridge_parameter;
#if 1
    Eigen::SelfAdjointEigenSolver<FMCA::Matrix> es;
    es.compute(Kloc, Eigen::ComputeEigenvectors);
    FMCA::Vector diag(locN);
    for (FMCA::Index i = 0; i < locN; ++i)
      diag(i) = es.eigenvalues()(i) > zeroTol
                    ? 1. / std::sqrt(es.eigenvalues()(i))
                    : 0;
    FMCA::Matrix invSqrt =
        es.eigenvectors() * diag.asDiagonal() * es.eigenvectors().transpose();
    FMCA::Vector col = invSqrt.col(pos);

#else
    Eigen::LLT<Matrix> llt;
    llt.compute(Kloc);
    FMCA::Vector rhs(Kloc.rows());
    rhs.setZero();
    rhs(pos) = 1;
    FMCA::Vector col = llt.matrixL().transpose().solve(rhs);
#endif
    std::vector<Eigen::Triplet<FMCA::Scalar>> local_triplets;
    for (FMCA::Index j = 0; j < locN; ++j)
      local_triplets.push_back(
          Eigen::Triplet<FMCA::Scalar>(i, inv_idcs[epsnn[i][j]], col(j)));
#pragma omp critical
    triplets.insert(triplets.end(), local_triplets.begin(),
                    local_triplets.end());
  }
  Eigen::SparseMatrix<FMCA::Scalar> invsqrtK(P.cols(), P.cols());
  invsqrtK.setFromTriplets(triplets.begin(), triplets.end());
  return invsqrtK;
}
}  // namespace FMCA
#endif
