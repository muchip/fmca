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
#ifndef FMCA_H2MATRIX_VOH2CLUSTERTREE_H_
#define FMCA_H2MATRIX_VOH2CLUSTERTREE_H_

namespace FMCA {

struct VOH2ClusterTreeNode : public H2ClusterTreeNodeBase<VOH2ClusterTreeNode> {
};

namespace internal {
template <typename ClusterTreeType>
struct traits<VOH2ClusterTree<ClusterTreeType>>
    : public traits<ClusterTreeType> {
  typedef VOH2ClusterTreeNode Node;
};
}  // namespace internal

/**
 *  \ingroup H2Matrix
 *  \brief The H2ClusterTree class manages the cluster bases for a given
 *         ClusterTree.
 *
 *         The tree structure from the ClusterTree is replicated here. This
 *         was a design decision as a cluster tree per se is not related to
 *         cluster bases. Also note that we just use pointers to clusters here.
 *         Thus, if the cluster tree is mutated or goes out of scope, we get
 *         dangeling pointers!
 */
template <typename ClusterTreeType>
class VOH2ClusterTree
    : public H2ClusterTreeBase<VOH2ClusterTree<ClusterTreeType>> {
 public:
  typedef typename internal::traits<VOH2ClusterTree>::Node Node;
  typedef H2ClusterTreeBase<VOH2ClusterTree<ClusterTreeType>> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::bb;
  using Base::block_id;
  using Base::derived;
  using Base::Es;
  using Base::indices;
  using Base::indices_begin;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
  using Base::V;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  VOH2ClusterTree() {}
  template <typename Moments, typename Derived>
  VOH2ClusterTree(const Moments &mom, const ClusterTreeBase<Derived> &ct) {
    init(mom, ct);
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename Moments, typename Derived>
  void init(const Moments &mom, const ClusterTreeBase<Derived> &ct) {
    std::vector<std::pair<const Derived *, VOH2ClusterTree *>> stack;
    // first copy the cluster tree
    stack.push_back(std::make_pair(std::addressof(ct.derived()), this));
    while (stack.size()) {
      // get current pair back from the stack
      const Derived *src = stack.back().first;
      VOH2ClusterTree *dest = stack.back().second;
      stack.pop_back();
      // copy the node
      (*dest).node().bb_ = (*src).node().bb_;
      (*dest).node().indices_ = (*src).derived().node().indices_;
      (*dest).node().indices_begin_ = (*src).derived().node().indices_begin_;
      (*dest).node().block_id_ = (*src).derived().node().block_id_;
      (*dest).node().block_size_ = (*src).derived().node().block_size_;
      // handle possible children by DFS
      if (src->nSons()) {
        dest->appendSons(src->nSons());
        for (Index i = 0; i < src->nSons(); ++i)
          stack.push_back(std::make_pair(std::addressof(src->sons(i)),
                                         std::addressof(dest->sons(i))));
      }
    }
    // init the cluster bases
    internal::compute_variable_order_cluster_bases_impl::compute(*this, mom);
  }
};

}  // namespace FMCA
#endif
