// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_WEDGELETS_WEDGELETTREE_H_
#define FMCA_WEDGELETS_WEDGELETTREE_H_

namespace FMCA {

struct WedgeletTreeNode : public WedgeletTreeNodeBase<WedgeletTreeNode> {};

namespace internal {
template <typename WedgeSplitter>
struct traits<WedgeletTree<WedgeSplitter>> {
  typedef WedgeletTreeNode Node;
  typedef ClusterSplitter::RandomProjection Splitter;
};
}  // namespace internal

/**
 *  \ingroup H2Matrix
 *  \brief The WedgeletTree class manages the cluster bases for a given
 *         ClusterTree.
 *
 *         The tree structure from the ClusterTree is replicated here. This
 *         was a design decision as a cluster tree per se is not related to
 *         cluster bases. Also note that we just use pointers to clusters here.
 *         Thus, if the cluster tree is mutated or goes out of scope, we get
 *         dangeling pointers!
 */
template <typename WedgeSplitter>
class WedgeletTree : public WedgeletTreeBase<WedgeletTree<WedgeSplitter>> {
 public:
  typedef typename internal::traits<WedgeletTree>::Node Node;
  typedef typename internal::traits<WedgeletTree>::Splitter Splitter;
  typedef WedgeletTreeBase<WedgeletTree<WedgeSplitter>> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::bb;
  using Base::block_id;
  using Base::derived;
  using Base::indices;
  using Base::indices_begin;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  WedgeletTree() {}
  WedgeletTree(const Matrix &P, const Matrix &F, const Index unif_splits = 4) {
    init(P, F, unif_splits);
  }
  //////////////////////////////////////////////////////////////////////////////
  void init(const Matrix &P, const Matrix &F, const Index unif_splits = 4) {
    // init cluster tree first
    using Initializer = internal::ClusterTreeInitializer<ClusterTree>;
    Splitter split;
    // set up root node
    Initializer::init_BoundingBox_impl(*this, 0, P);
    (*this).node().indices_begin_ = 0;
    (*this).node().indices_ = std::shared_ptr<Index>(
        new Index[P.cols()], std::default_delete<Index[]>());
    (*this).node().block_size_ = P.cols();
    Index *indices = (*this).node().indices_.get();
    for (Index i = 0; i < (*this).block_size(); ++i) indices[i] = i;

    // use a DFS to construct adaptive wedgelet tree
    std::vector<WedgeletTree *> queue;
    queue.push_back(this);
    while (queue.size()) {
      WedgeletTree &wt = *(queue.back());
      queue.pop_back();
      if (wt.level() < unif_splits) {
        wt.appendSons(2);
        for (Index i = 0; i < 2; ++i) {
          wt.sons(i).node().bb_ = wt.node().bb_;
          wt.sons(i).node().indices_ = wt.node().indices_;
          wt.sons(i).node().block_size_ = wt.node().block_size_;
          wt.sons(i).node().indices_begin_ = wt.node().indices_begin_;
          queue.push_back(std::addressof(wt.sons(i)));
        }
        // split index set and set sons bounding boxes
        split(P, wt.sons(0).node(), wt.sons(1).node());
        if (wt.level() == unif_splits - 1) {
          // compute representation
        }
      } else {
        // compute representation

        // should I split?
        if (split) {
        }
      }
    }
  }
};

}  // namespace FMCA
#endif
