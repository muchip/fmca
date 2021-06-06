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
#ifndef FMCA_SAMPLETS_BIVARIATECOMPRESSOR_H_
#define FMCA_SAMPLETS_BIVARIATECOMPRESSOR_H_

namespace FMCA {

template <typename SampletTree> class BivariateCompressor {
public:
  typedef typename SampletTree::value_type value_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  BivariateCompressor(){};
  template <typename Functor>
  BivariateCompressor(const Eigen::Matrix<value_type, SampletTree::dimension,
                                          Eigen::Dynamic> &P,
                      const SampletTree &ST, const Functor &fun,
                      value_type a_param = 2., value_type dprime = 1.,
                      value_type operator_order = 0.) {
    init(P, ST, fun, a_param, dprime, operator_order);
  }

  template <typename Functor>
  void init(const Eigen::Matrix<value_type, SampletTree::dimension,
                                Eigen::Dynamic> &P,
            const SampletTree &ST, const Functor &fun, value_type a_param = 2.,
            value_type dprime = 1., value_type op = 0.) {
    a_param_ = a_param;
    dprime_ = dprime;
    op_ = op;
    J_param_ = ST.tree_data_->max_wlevel_;
    dtilde_ = ST.tree_data_->dtilde_;
    cut_const1_ = J_param_ * (dprime_ - op_) / (dtilde_ + op_);
    cut_const2_ = 0.5 * (dprime_ + dtilde_) / (dtilde_ + op_);
    geo_diam_ = ST.cluster_->get_tree_data().geometry_diam_;
    triplet_list_.clear();
    buffer_.clear();
    buffer_.resize(ST.tree_data_->samplet_list.size());
    setupColumn(P, ST, ST, fun);
    // set up remainder of the first column
    for (auto i = 1; i < ST.tree_data_->samplet_list.size(); ++i) {
      const IndexType block_id = ST.tree_data_->samplet_list[i]->block_id_;
      const IndexType start_index =
          ST.tree_data_->samplet_list[i]->start_index_;
      const IndexType nsamplets = ST.tree_data_->samplet_list[i]->nsamplets_;
      const IndexType nscalfs = ST.tree_data_->samplet_list[i]->nscalfs_;
      auto it = buffer_[block_id].find(ST.block_id_);
      eigen_assert(it != buffer_[block_id].end() &&
                   "there is a missing root block!");
      for (auto k = 0; k < ST.Q_.cols(); ++k)
        for (auto j = 0; j < nsamplets; ++j)
          triplet_list_.push_back(
              Eigen::Triplet<value_type>(start_index + j, ST.start_index_ + k,
                                         (it->second)(nscalfs + j, k)));
      buffer_[block_id].erase(it);
    }
    // for (auto i = 0; i < ST.tree_data_->samplet_list.size(); ++i)
    //  assemblePattern(*(ST.tree_data_->samplet_list[i]), ST, triplet_list_);
    std::cout << "a:   " << a_param_ << std::endl;
    std::cout << "dp:  " << dprime_ << std::endl;
    std::cout << "dt:  " << dtilde_ << std::endl;
    std::cout << "op:  " << op_ << std::endl;
    std::cout << "J:   " << J_param_ << std::endl;
    std::cout << "cc1: " << cut_const1_ << std::endl;
    std::cout << "cc2: " << cut_const2_ << std::endl;
    std::cout << "diam:" << geo_diam_ << std::endl;
  }

  const std::vector<Eigen::Triplet<value_type>> &get_Pattern_triplets() const {
    return triplet_list_;
  }

  eigenMatrix get_S(IndexType dim) const {
    Eigen::SparseMatrix<value_type> S(dim, dim);
    S.setFromTriplets(triplet_list_.begin(), triplet_list_.end());
    return eigenMatrix(S);
  }
  //////////////////////////////////////////////////////////////////////////////
  /// cutOff criterion
  //////////////////////////////////////////////////////////////////////////////
  value_type cutOffParameter(IndexType j, IndexType jp) {
    const value_type first = j < jp ? 1. / (1 << j) : 1. / (1 << jp);
    const value_type second =
        std::pow(2., cut_const1_ - (j + jp) * cut_const2_);
    return a_param_ * first * geo_diam_; //(first > second ? first : second);
  }
  //////////////////////////////////////////////////////////////////////////////
  bool cutOff(IndexType j, IndexType jp, value_type dist) {
    return dist > cutOffParameter(j, jp);
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief recursively computes for a given pair of row and column clusters
   *         the four blocks [A^PhiPhi, A^PhiPsi; A^PsiPhi, A^PsiPsi]
   **/
  template <typename Functor>
  eigenMatrix recursivelyComputeBlock(
      const Eigen::Matrix<value_type, SampletTree::dimension, Eigen::Dynamic>
          &P,
      eigenMatrix *S, const SampletTree &TR, const SampletTree &TC,
      const Functor &fun) {
    eigenMatrix buf(0, 0);
    eigenMatrix retval(0, 0);
    if (!TR.sons_.size() && !TC.sons_.size()) {
      // both are leafs: compute the block and return
      buf.resize(TR.cluster_->get_indices().size(),
                 TC.cluster_->get_indices().size());
      for (auto j = 0; j < TC.cluster_->get_indices().size(); ++j)
        for (auto i = 0; i < TR.cluster_->get_indices().size(); ++i)
          buf(i, j) = fun(P.col(TR.cluster_->get_indices()[i]),
                          P.col(TC.cluster_->get_indices()[j]));
      retval = TR.Q_.transpose() * buf * TC.Q_;
    } else if (!TR.sons_.size() && TC.sons_.size()) {
      // the row cluster is a leaf cluster: recursion on the col cluster
      for (auto j = 0; j < TC.sons_.size(); ++j) {
        eigenMatrix ret = recursivelyComputeBlock(P, S, TR, TC.sons_[j], fun);
        buf.conservativeResize(ret.rows(), buf.cols() + TC.sons_[j].nscalfs_);
        buf.rightCols(TC.sons_[j].nscalfs_) =
            ret.leftCols(TC.sons_[j].nscalfs_);
      }
      retval = buf * TC.Q_;
    } else if (TR.sons_.size() && !TC.sons_.size()) {
      // the col cluster is a leaf cluster: recursion on the row cluster
      for (auto i = 0; i < TR.sons_.size(); ++i) {
        eigenMatrix ret = recursivelyComputeBlock(P, S, TR.sons_[i], TC, fun);
        buf.conservativeResize(ret.cols(), buf.cols() + TR.sons_[i].nscalfs_);
        buf.rightCols(TR.sons_[i].nscalfs_) =
            ret.transpose().leftCols(TR.sons_[i].nscalfs_);
      }
      retval = (buf * TR.Q_).transpose();
    } else {
      // neither is a leaf, let recursion handle this
      for (auto i = 0; i < TR.sons_.size(); ++i) {
        eigenMatrix ret1(0, 0);
        for (auto j = 0; j < TC.sons_.size(); ++j) {
          eigenMatrix ret2 =
              recursivelyComputeBlock(P, S, TR.sons_[i], TC.sons_[j], fun);
          ret1.conservativeResize(ret2.rows(),
                                  ret1.cols() + TC.sons_[j].nscalfs_);
          ret1.rightCols(TC.sons_[j].nscalfs_) =
              ret2.leftCols(TC.sons_[j].nscalfs_);
        }
        ret1 = ret1 * TC.Q_;
        buf.conservativeResize(ret1.cols(), buf.cols() + TR.sons_[i].nscalfs_);
        buf.rightCols(TR.sons_[i].nscalfs_) =
            ret1.transpose().leftCols(TR.sons_[i].nscalfs_);
      }
      retval = (buf * TR.Q_).transpose();
    }
#if 0
    if (TR.nsamplets_ && TC.nsamplets_) {
      IndexType offset = 0;
      if (!TC.wlevel_ && !TR.wlevel_)
        offset = TR.nscalfs_;
      if ((S->block(TR.start_index_ + offset, TC.start_index_ + offset,
                    TR.nsamplets_, TC.nsamplets_) -
           retval.block(TR.nscalfs_, TC.nscalfs_, TR.nsamplets_, TC.nsamplets_))
              .norm() > 1e-13) {
        std::cout << TR.block_id_ << " " << TC.block_id_ << " <--\n";
        std::cout << S->block(TR.start_index_ + offset,
                              TC.start_index_ + offset, TR.nsamplets_,
                              TC.nsamplets_)
                  << "\n---------\n";
        std::cout << retval.block(TR.nscalfs_, TC.nscalfs_, TR.nsamplets_,
                                  TC.nsamplets_)
                  << "\n******************\n";
      }
    }
#endif
    return retval;
  }

private:
  //////////////////////////////////////////////////////////////////////////////
  value_type computeDistance(const SampletTree &TR, const SampletTree &TC) {
    const value_type row_radius =
        0.5 *
        (TR.cluster_->get_bb().col(0) - TR.cluster_->get_bb().col(1)).norm();
    const value_type col_radius =
        0.5 *
        (TC.cluster_->get_bb().col(0) - TC.cluster_->get_bb().col(1)).norm();
    const value_type dist =
        0.5 * (TR.cluster_->get_bb().col(0) - TC.cluster_->get_bb().col(0) +
               TR.cluster_->get_bb().col(1) - TC.cluster_->get_bb().col(1))
                  .norm() -
        row_radius - col_radius;
    return dist > 0 ? dist : 0;
  } //////////////////////////////////////////////////////////////////////////////
  template <typename Functor>
  void setupRow(const Eigen::Matrix<value_type, SampletTree::dimension,
                                    Eigen::Dynamic> &P,
                const SampletTree &TR, const SampletTree &TC,
                const Functor &fun) {
    eigenMatrix block(0, 0);
    eigenMatrix buf(0, 0);
    ////////////////////////////////////////////////////////////////////////////
    // if there are children of the row cluster, we proceed recursively
    if (TR.sons_.size()) {
      for (auto i = 0; i < TR.sons_.size(); ++i) {
        const value_type dist = computeDistance(TR.sons_[i], TC);
        if (!cutOff(TR.sons_[i].wlevel_, TC.wlevel_, dist)) {
          setupRow(P, TR.sons_[i], TC, fun);
          auto it = buffer_[TR.sons_[i].block_id_].find(TC.block_id_);
          eigen_assert(it != buffer_[TR.sons_[i].block_id_].end() &&
                       "entry should exist row");
          buf.conservativeResize((it->second).cols(),
                                 buf.cols() + TR.sons_[i].nscalfs_);
          buf.rightCols(TR.sons_[i].nscalfs_) =
              (it->second).transpose().leftCols(TR.sons_[i].nscalfs_);
          if (it->first != 0)
            buffer_[TR.sons_[i].block_id_].erase(it);
        } else {
          eigenMatrix ret =
              recursivelyComputeBlock(P, nullptr, TR.sons_[i], TC, fun);
          buf.conservativeResize(ret.cols(), buf.cols() + TR.sons_[i].nscalfs_);
          buf.rightCols(TR.sons_[i].nscalfs_) =
              ret.transpose().leftCols(TR.sons_[i].nscalfs_);
        }
      }
      block = (buf * TR.Q_).transpose();
      ////////////////////////////////////////////////////////////////////////////
      // if we are at a leaf of the row cluster tree, we compute the entire row
    } else {
      // if TR and TC are both leafs, we compute the corresponding matrix block
      if (!TC.sons_.size())
        block = recursivelyComputeBlock(P, nullptr, TR, TC, fun);
      // if TC is not a leaf, we proceed by assembling the remainder of the row
      else {
        for (auto j = 0; j < TC.sons_.size(); ++j) {
          auto it = buffer_[TR.block_id_].find(TC.sons_[j].block_id_);
          if (it != buffer_[TR.block_id_].end()) {
            buf.conservativeResize((it->second).rows(),
                                   buf.cols() + TC.sons_[j].nscalfs_);
            buf.rightCols(TC.sons_[j].nscalfs_) =
                (it->second).leftCols(TC.sons_[j].nscalfs_);
          } else {
            eigenMatrix ret =
                recursivelyComputeBlock(P, nullptr, TR, TC.sons_[j], fun);
            buf.conservativeResize(ret.rows(),
                                   buf.cols() + TC.sons_[j].nscalfs_);
            buf.rightCols(TC.sons_[j].nscalfs_) =
                ret.leftCols(TC.sons_[j].nscalfs_);
          }
        }
        block = buf * TC.Q_;
      }
      for (auto j = 0; j < TC.sons_.size(); ++j)
        buffer_[TR.block_id_].erase(TC.sons_[j].block_id_);
    }
    if (TR.nsamplets_ && TC.nsamplets_ && TC.wlevel_ && TR.wlevel_) {
      for (auto k = 0; k < TC.nsamplets_; ++k)
        for (auto j = 0; j < TR.nsamplets_; ++j)
          triplet_list_.push_back(Eigen::Triplet<value_type>(
              TR.start_index_ + j, TC.start_index_ + k,
              block(TR.nscalfs_ + j, TC.nscalfs_ + k)));
    }
    buffer_[TR.block_id_].emplace(std::make_pair(TC.block_id_, block));
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  template <typename Functor>
  void setupColumn(const Eigen::Matrix<value_type, SampletTree::dimension,
                                       Eigen::Dynamic> &P,
                   const SampletTree &TR, const SampletTree &TC,
                   const Functor &fun) {
    Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> retval;

    if (TC.sons_.size())
      for (auto i = 0; i < TC.sons_.size(); ++i)
        setupColumn(P, TR, TC.sons_[i], fun);
    setupRow(P, TR, TC, fun);
    auto it = buffer_[TR.block_id_].find(TC.block_id_);
    eigen_assert(it != buffer_[TR.block_id_].end() &&
                 "there is a missing root block!");
    if (!TC.wlevel_)
      for (auto k = 0; k < TC.Q_.cols(); ++k)
        for (auto j = 0; j < TR.Q_.cols(); ++j)
          triplet_list_.push_back(Eigen::Triplet<value_type>(
              TR.start_index_ + j, TC.start_index_ + k, (it->second)(j, k)));
    else if (TC.nsamplets_)
      for (auto k = 0; k < TC.nsamplets_; ++k)
        for (auto j = 0; j < TR.Q_.cols(); ++j)
          triplet_list_.push_back(Eigen::Triplet<value_type>(
              TR.start_index_ + j, TC.start_index_ + k,
              (it->second)(j, TC.nscalfs_ + k)));
    buffer_[TR.block_id_].erase(it);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void assemblePattern(const SampletTree &TR, const SampletTree &TC,
                       std::vector<Eigen::Triplet<value_type>> &triplet_list) {
    const value_type dist = computeDistance(TR, TC);
    // if either cluster is empty or the cutoff is satisfied, return
    if (!TR.cluster_->get_indices().size() ||
        !TC.cluster_->get_indices().size() ||
        cutOff(TR.wlevel_, TC.wlevel_, dist))
      return;
    // add matrix entry if there is something to add
    if ((!TR.wlevel_ || TR.nsamplets_) && (!TC.wlevel_ || TC.nsamplets_)) {
      triplet_list.push_back(
          Eigen::Triplet<value_type>(TR.block_id_, TC.block_id_, dist));
    }
    // check children
    for (auto j = 0; j < TC.sons_.size(); ++j) {
      assemblePattern(TR, TC.sons_[j], triplet_list);
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<value_type>> triplet_list_;
  std::vector<std::map<IndexType, eigenMatrix>> buffer_;
  value_type a_param_;
  value_type dprime_;
  value_type dtilde_;
  value_type J_param_;
  value_type op_;
  value_type cut_const1_;
  value_type cut_const2_;
  value_type geo_diam_;
  IndexType max_wlevel_;
  IndexType fun_calls_;
}; // namespace FMCA
} // namespace FMCA
#endif
