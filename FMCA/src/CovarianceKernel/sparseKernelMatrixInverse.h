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
#ifndef FMCA_COVARIANCEKERNEL_SPARSEKERNELMATRIXINVERSE_H_
#define FMCA_COVARIANCEKERNEL_SPARSEKERNELMATRIXINVERSE_H_

namespace FMCA {

template <typename Derived>
Eigen::SparseMatrix<Scalar> sparseKernelMatrixInverse(
    const CovarianceKernel &K, const ClusterTreeBase<Derived> &CT,
    const Matrix &P, const Index fps = 1, const Scalar ridge_parameter = 0) {
  const Vector mdv = minDistanceVector(CT, P);
  const Scalar fill_distance = mdv.maxCoeff();
  const Scalar search_radius =
      fps * fill_distance * std::abs(std::log(fill_distance));
  std::cout << std::string(30, '-') << "sparseInverse" << std::string(30, '-')
            << std::endl;
  std::cout << "fill distance:                " << fill_distance << std::endl;
  std::cout << "search radius:                " << search_radius << std::endl;
  // compute epsilon nearest neighbours for all points
  std::vector<std::vector<Index>> epsnn(P.cols());
#pragma omp parallel for
  for (Index i = 0; i < epsnn.size(); ++i)
    epsnn[i] = epsNN(CT, P, P.col(i), search_radius);
  // evaluate localized inverse
  std::vector<Eigen::Triplet<Scalar>> triplets;
  // compute permutation from original order to cluster order
  std::vector<Index> inv_idcs(P.cols());
  for (FMCA::Index i = 0; i < inv_idcs.size(); ++i)
    inv_idcs[CT.indices()[i]] = i;
    // actually compute the localized inverse
#pragma omp parallel for
  for (Index i = 0; i < epsnn.size(); ++i) {
    const Index locN = epsnn[i].size();
    Matrix Ploc(P.rows(), locN);
    Index pos = 0;
    for (Index j = 0; j < locN; ++j) {
      Ploc.col(j) = P.col(epsnn[i][j]);
      pos = epsnn[i][j] == i ? j : pos;
    }
    Matrix Kloc = K.eval(Ploc, Ploc);
    Kloc.diagonal().array() += ridge_parameter;
    Vector rhs(locN);
    rhs.setZero();
    rhs(pos) = 1;
    Vector col = Kloc.ldlt().solve(rhs);
    std::vector<Eigen::Triplet<Scalar>> local_triplets;
    for (Index j = 0; j < locN; ++j)
      local_triplets.push_back(
          Eigen::Triplet<Scalar>(inv_idcs[i], inv_idcs[epsnn[i][j]], col(j)));
#pragma omp critical
    triplets.insert(triplets.end(), local_triplets.begin(),
                    local_triplets.end());
  }
  Eigen::SparseMatrix<FMCA::Scalar> invK(P.cols(), P.cols());
  invK.setFromTriplets(triplets.begin(), triplets.end());
  return invK;
}
}  // namespace FMCA
#endif
