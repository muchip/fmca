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
  typedef ClusterSplitter::CardinalityBisection Splitter;
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
  WedgeletTree(const Matrix &P, const Matrix &F, const Index q = 0,
               const Index unif_splits = 4) {
    init(P, F, unif_splits);
  }
  //////////////////////////////////////////////////////////////////////////////
  void init(const Matrix &P, const Matrix &F, const Index q = 0,
            const Index unif_splits = 5, const Scalar tol = 1e-2) {
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

    std::priority_queue<std::pair<Scalar, WedgeletTree *>> tree_leafs;
    // use a DFS to construct inital tree
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

      } else if (wt.block_size()) {
        // determine maximum possible polynomial degree
        wt.node().dim_ = P.rows();
        wt.node().deg_ = q;
        while (binomialCoefficient(wt.node().dim_ + wt.node().deg_,
                                   wt.node().deg_) > wt.block_size())
          wt.node().deg_ -= 1;
        // compute least square approximation in the cluster
        MultiIndexSet<TotalDegree> idcs(wt.node().dim_, wt.node().deg_);
        Scalar err = 0;
        wt.node().C_ = WedgeletTreeHelper::computeLeastSquaresFit(
            wt.indices(), wt.block_size(), P, F, idcs, &err);
        // store error and leaf in the heap
        tree_leafs.push(std::make_pair(err, std::addressof(wt)));
        // generate fake landmark index
        FMCA::Vector mean(P.rows());
        for (Index i = 0; i < wt.block_size(); ++i)
          mean += P.col(wt.indices()[i]);
        mean *= 1. / wt.block_size();
        Scalar min_dist = FMCA_INF;
        for (Index i = 0; i < wt.block_size(); ++i) {
          const Scalar dist = (mean - P.col(wt.indices()[i])).norm();
          if (dist < min_dist) {
            wt.node().landmark_ = wt.indices()[i];
            min_dist = dist;
          }
        }
      }
    }
    std::cout << "uniform refinement computed" << std::endl;
    // now adaptively refine the leaf with the biggest error
    while (tree_leafs.size() && tree_leafs.top().first > tol * tol) {
      WedgeletTree &wt = *(tree_leafs.top().second);
      Scalar err = tree_leafs.top().first;
      std::cout << "node: " << tree_leafs.top().second << "err: " << err
                << std::endl;
      tree_leafs.pop();
      Scalar split_error = FMCA_INF;
      Matrix C1;
      Matrix C2;
      Scalar err1 = 0;
      Scalar err2 = 0;
      Index q1 = 0;
      Index q2 = 0;
      std::vector<Index> c1;
      std::vector<Index> c2;
      Index lm2;
      const Index lm1 = wt.node().landmark_;
#pragma omp parallel for schedule(dynamic)
      for (Index i = 0; i < wt.block_size(); ++i)
        if (wt.indices()[i] != lm1) {
          const Index cur_lm2 = wt.indices()[i];
          std::vector<Index> cur_c1;
          std::vector<Index> cur_c2;
          cur_c1.reserve(wt.block_size());
          cur_c2.reserve(wt.block_size());
          Matrix cur_C1;
          Matrix cur_C2;
          Scalar cur_err1 = FMCA_INF;
          Scalar cur_err2 = FMCA_INF;
          Index cur_q1 = wt.node().deg_;
          Index cur_q2 = wt.node().deg_;

          for (Index j = 0; j < wt.block_size(); ++j) {
            const Scalar dist1 = (P.col(lm1) - P.col(wt.indices()[j])).norm();
            const Scalar dist2 =
                (P.col(cur_lm2) - P.col(wt.indices()[j])).norm();
            if (dist1 < dist2)
              cur_c1.push_back(wt.indices()[j]);
            else
              cur_c2.push_back(wt.indices()[j]);
          }
          if (cur_c1.size()) {
            while (binomialCoefficient(wt.node().dim_ + cur_q1, cur_q1) >
                   cur_c1.size())
              cur_q1 -= 1;
            MultiIndexSet<TotalDegree> idcs(wt.node().dim_, cur_q1);
            cur_C1 = WedgeletTreeHelper::computeLeastSquaresFit(
                cur_c1.data(), cur_c1.size(), P, F, idcs, &cur_err1);
          }
          if (cur_c2.size()) {
            while (binomialCoefficient(wt.node().dim_ + cur_q2, cur_q2) >
                   cur_c2.size())
              cur_q2 -= 1;
            MultiIndexSet<TotalDegree> idcs(wt.node().dim_, cur_q2);
            cur_C2 = WedgeletTreeHelper::computeLeastSquaresFit(
                cur_c2.data(), cur_c2.size(), P, F, idcs, &cur_err2);
          }
#pragma omp critical
          {
            if (cur_err1 + cur_err2 < split_error) {
              split_error = cur_err1 + cur_err2;
              err1 = cur_err1;
              err2 = cur_err2;
              C1 = cur_C1;
              C2 = cur_C2;
              lm2 = cur_lm2;
              c1 = cur_c1;
              c2 = cur_c2;
              q1 = cur_q1;
              q2 = cur_q2;
            }
          }
        }
      // we have now determined the best possible split, split the leave if it
      // reduces the error
      std::cout << err1 << " " << err2 << " " << wt.block_size() << " "
                << c1.size() << " " << c2.size() << " " << wt.level()
                << std::endl;

      if (c1.size() && c2.size()) {
        wt.appendSons(2);
        for (Index i = 0; i < 2; ++i) {
          wt.sons(i).node().bb_ = wt.node().bb_;
          wt.sons(i).node().indices_ = wt.node().indices_;
          wt.sons(i).node().block_size_ = wt.node().block_size_;
          wt.sons(i).node().indices_begin_ = wt.node().indices_begin_;
          wt.sons(0).node().dim_ = wt.node().dim_;
        }
        wt.sons(0).node().deg_ = q1;
        wt.sons(1).node().deg_ = q2;
        wt.sons(0).node().C_ = C1;
        wt.sons(1).node().C_ = C2;
        wt.sons(0).node().landmark_ = lm1;
        wt.sons(1).node().landmark_ = lm2;
        wt.sons(0).node().block_size_ = c1.size();
        wt.sons(1).node().block_size_ = c2.size();
        wt.sons(1).node().indices_begin_ = wt.node().indices_begin_ + c1.size();
        {
          Index *idcs = wt.node().indices_.get();
          Index offs = wt.node().indices_begin_;
          for (Index i = 0; i < c1.size(); ++i) idcs[offs + i] = c1[i];
          offs += c1.size();
          for (Index i = 0; i < c2.size(); ++i) idcs[offs + i] = c2[i];
        }
        tree_leafs.push(std::make_pair(err1, std::addressof(wt.sons(0))));
        tree_leafs.push(std::make_pair(err2, std::addressof(wt.sons(1))));
      }
    }
  }
};

}  // namespace FMCA
#endif
