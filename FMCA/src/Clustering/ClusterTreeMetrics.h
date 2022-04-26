// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_CLUSTERING_CLUSTERTREEMETRICS_H_
#define FMCA_CLUSTERING_CLUSTERTREEMETRICS_H_

namespace FMCA {

enum Admissibility { Refine = 0, LowRank = 1, Dense = 2 };

template <typename Derived>
typename Derived::eigenMatrix::Scalar computeDistance(
    const ClusterTreeBase<Derived> &TR, const ClusterTreeBase<Derived> &TC) {
  typedef typename Derived::eigenMatrix::Scalar value_type;

  const value_type row_radius = 0.5 * TR.bb().col(2).norm();
  const value_type col_radius = 0.5 * TC.bb().col(2).norm();
  const value_type dist =
      0.5 * (TR.bb().col(0) - TC.bb().col(0) + TR.bb().col(1) - TC.bb().col(1))
                .norm() -
      row_radius - col_radius;
  return dist > 0 ? dist : 0;
}

template <typename Derived>
Admissibility compareCluster(const ClusterTreeBase<Derived> &cluster1,
                             const ClusterTreeBase<Derived> &cluster2,
                             typename Derived::eigenMatrix::Scalar eta) {
  typedef typename Derived::eigenMatrix::Scalar value_type;
  Admissibility retval;
  const value_type dist = computeDistance(cluster1, cluster2);
  const value_type row_radius = 0.5 * cluster1.bb().col(2).norm();
  const value_type col_radius = 0.5 * cluster2.bb().col(2).norm();
  const value_type radius = row_radius > col_radius ? row_radius : col_radius;

  if (radius > eta * dist) {
    // check if either cluster is a leaf in that case,
    // compute the full matrix block
    if (!cluster1.nSons() || !cluster2.nSons())
      return Dense;
    else
      return Refine;
  } else
    return LowRank;
}

}  // namespace FMCA
#endif
