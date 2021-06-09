// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_H2MATRIX_H2CLUSTERTREE_H_
#define FMCA_H2MATRIX_H2CLUSTERTREE_H_

namespace FMCA {

/**
 *  \ingroup H2Matrix
 *  \brief The H2ClusterTree class manages cluster trees for H2Matrices in
 *         arbitrary dimension. It is derived from the ClusterTree class
 */
template <typename ValueType, IndexType Dim, IndexType MinClusterSize,
          IndexType Deg = 3,
          typename Splitter =
              ClusterSplitter::CardinalityBisection<ValueType, Dim>>
class H2ClusterTree
    : public ClusterTree<ValueType, Dim, MinClusterSize, Splitter> {
  friend Splitter;
  using CT = ClusterTree<ValueType, Dim, MinClusterSize, Splitter>;

public:
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  H2ClusterTree() {}
  H2ClusterTree(const Eigen::Matrix<ValueType, Dim, Eigen::Dynamic> &P) {
    init(P);
  }

  void init(const Eigen::Matrix<ValueType, Dim, Eigen::Dynamic> &P) {
    // init underlying tree structure
    CT::init(P);
    // set up the tensor product interpolator
    TP_interp_ =
        std::make_shared<TensorProductInterpolator<ValueType, Dim, Deg>>();
    TP_interp_->init();
    // now compute the H2 cluster bases
    computeClusterBases(P);
  }

  void
  computeClusterBases(const Eigen::Matrix<ValueType, Dim, Eigen::Dynamic> &P) {
    if (CT::sons_.size()) {
      for (auto i = 0; i < CT::sons_.size(); ++i)
        CT::sons_[i].computeClusterBases(P);
      // compute transfer matrices
    } else {
      V_.resize(TP_interp_->get_Xi().cols(), CT::get_indices().size());
    }
  }
  //////////////////////////////////////////////////////////////////////////////
private:
  std::shared_ptr<TensorProductInterpolator<ValueType, Dim, Deg>> TP_interp_;
  std::vector<Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>> E_;
  Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> V_;
};

} // namespace FMCA
#endif
