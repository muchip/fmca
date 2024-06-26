// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2024, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_SAMPLETS_SAMPLETADAPTIVTREESEARCH_H_
#define FMCA_SAMPLETS_SAMPLETADAPTIVTREESEARCH_H_

namespace FMCA {

namespace internal {
template <typename T>
struct CompIndexSort {
  CompIndexSort(const T &field) : field_(field){};

  template <typename R, typename S>
  bool operator()(const R &i, const S &j) {
    return field_[i] > field_[j];
  }
  const T &field_;
};
}  // namespace internal

/**
 *  \ingroup Samplets
 *  \brief implements the second Binev-DeVore algorithm for adaptive tree
 *         search
 **/

template <typename Derived>
std::vector<const Derived *> adaptiveTreeSearch(
    const SampletTreeBase<Derived> &st, const Vector tdata,
    const Scalar thres) {
  const Index nclusters = std::distance(st.begin(), st.end());
  std::vector<const Derived *> retval(nclusters);
  std::vector<const Derived *> cluster_map(nclusters);
  Vector e(nclusters);
  Vector q(nclusters);
  Vector etilde(nclusters);
  e.setZero();
  q.setZero();
  etilde.setZero();
  // serialize the tree first
  for (const auto &it : st) cluster_map[it.block_id()] = std::addressof(it);

  // set up e functional (bottom up tree traversal)
  for (auto it = cluster_map.rbegin(); it != cluster_map.rend(); ++it) {
    const Derived &node = **it;
    const Index ndist = node.is_root() ? node.Q().cols() : node.nsamplets();
    e(node.block_id()) = tdata.segment(node.start_index(), ndist).squaredNorm();
    if (node.nSons()) {
      // set up q as the sum of the children's energies. proper scaling is
      // performed in the next traversal
      for (Index i = 0; i < node.nSons(); ++i)
        q(node.block_id()) += e(node.sons(i).block_id());
      // update the clusters functional by the values of the children's
      // functionals
      e(node.block_id()) += q(node.block_id());
    }
  }
  std::cout << " " << e(0) << " " << tdata.squaredNorm() << std::endl;
  // set up etilde functional (top down tree traversal)
  for (auto it = cluster_map.begin(); it != cluster_map.end(); ++it) {
    const Derived &node = **it;
    // assign root cluster
    if (node.is_root()) etilde(node.block_id()) = e(node.block_id());
    // update q (either it is root or etilde was set by the parent)
    q(node.block_id()) *=
        etilde(node.block_id()) /
        (e(node.block_id()) + etilde(node.block_id()) + FMCA_ZERO_TOLERANCE);
    // set etilde for the children
    for (Index i = 0; i < node.nSons(); ++i)
      etilde(node.sons(i).block_id()) = q(node.block_id());
  }

  // sort etilde and add clusters due to their priority
  std::vector<Index> block_ids(nclusters);
  std::iota(block_ids.begin(), block_ids.end(), 0);
  std::sort(block_ids.begin(), block_ids.end(),
            internal::CompIndexSort<Vector>(etilde));

  const Scalar total_Etilde = etilde.sum();
  Scalar current_Etilde = 0;
  Index nnz = 0;
  while (total_Etilde - current_Etilde >= thres) {
    current_Etilde += etilde(block_ids[nnz]);
    retval[block_ids[nnz]] = cluster_map[block_ids[nnz]];
    ++nnz;
  }
  std::cout << "total energy in tree:         " << total_Etilde << std::endl;
  std::cout << "total number of clusters:     " << nclusters << std::endl;
  std::cout << "clusters in adaptive tree:    " << nnz << std::endl;
  // add also non present children
  for (Index i = 0; i < retval.size(); ++i)
    if (retval[i] != nullptr) {
      bool active_child = false;
      for (Index j = 0; j < retval[i]->nSons(); ++j)
        if (retval[retval[i]->sons(j).block_id()] != nullptr)
          active_child = true;
      if (active_child)
        for (Index j = 0; j < retval[i]->nSons(); ++j)
          retval[retval[i]->sons(j).block_id()] =
              cluster_map[retval[i]->sons(j).block_id()];
    }
  return retval;
}

}  // namespace FMCA
#endif
