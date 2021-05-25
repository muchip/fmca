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
  BivariateCompressor(){};
  template <typename Functor>
  BivariateCompressor(const SampletTree &ST, const Functor &fun,
                      value_type a_param = 2., value_type dprime = 1.,
                      value_type operator_order = 0.) {
    init(ST, fun, a_param, dprime, operator_order);
  }

  template <typename Functor>
  void init(const SampletTree &ST, const Functor &fun, value_type a_param = 2.,
            value_type dprime = 1., value_type op = 0.) {
    a_param_ = a_param;
    dprime_ = dprime;
    op_ = op;
    J_param_ = ST.tree_data_->max_wlevel_;
    dtilde_ = ST.tree_data_->dtilde_;
    cut_const1_ = J_param_ * (dprime_ - op_) / (dtilde_ + op_);
    cut_const2_ = 0.5 * (dprime_ + dtilde_) / (dtilde_ + op_);
    triplet_list_.clear();
    for (auto i = 0; i < ST.tree_data_->samplet_list.size(); ++i)
      assemblePattern(*(ST.tree_data_->samplet_list[i]), ST, triplet_list_);

    std::cout << "a:   " << a_param_ << std::endl;
    std::cout << "dp:  " << dprime_ << std::endl;
    std::cout << "dt:  " << dtilde_ << std::endl;
    std::cout << "op:  " << op_ << std::endl;
    std::cout << "J:   " << J_param_ << std::endl;
    std::cout << "cc1: " << cut_const1_ << std::endl;
    std::cout << "cc2: " << cut_const2_ << std::endl;
  }
  const std::vector<Eigen::Triplet<value_type>> &get_Pattern_triplets() const {
    return triplet_list_;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// cutOff criterion
  //////////////////////////////////////////////////////////////////////////////
  value_type cutOffParameter(IndexType j, IndexType jp) {
    const value_type first = j < jp ? 1. / (1 << j) : 1. / (1 << jp);
    const value_type second =
        std::pow(2., cut_const1_ - (j + jp) * cut_const2_);
    return a_param_ * first; // (first > second ? first : second);
  }
  //////////////////////////////////////////////////////////////////////////////
  bool cutOff(IndexType j, IndexType jp, value_type dist) {
    return dist > cutOffParameter(j, jp);
  }

private:
  void assemblePattern(const SampletTree &rowTree, const SampletTree &colTree,
                       std::vector<Eigen::Triplet<value_type>> &triplet_list) {
    // set distance to the distance of the midpoints
    const value_type row_radius = 0.5 * (rowTree.cluster_->get_bb().col(0) -
                                         rowTree.cluster_->get_bb().col(1))
                                            .norm();
    const value_type col_radius = 0.5 * (colTree.cluster_->get_bb().col(0) -
                                         colTree.cluster_->get_bb().col(1))
                                            .norm();
    const value_type dist = 0.5 * (rowTree.cluster_->get_bb().col(0) -
                                   colTree.cluster_->get_bb().col(0) +
                                   rowTree.cluster_->get_bb().col(1) -
                                   colTree.cluster_->get_bb().col(1))
                                      .norm() -
                            row_radius - col_radius;
    // if either cluster is empty or the cutoff is satisfied, return
    if (!rowTree.cluster_->get_indices().size() ||
        !colTree.cluster_->get_indices().size() ||
        cutOff(rowTree.wlevel_, colTree.wlevel_, dist))
      return;
    // add matrix entry if there is something to add
    if ((!rowTree.wlevel_ || rowTree.Q_W_.size()) &&
        (!colTree.wlevel_ || colTree.Q_W_.size())) {
      triplet_list.push_back(Eigen::Triplet<value_type>(
          rowTree.block_id_, colTree.block_id_, dist));
    }
    // check children
    for (auto j = 0; j < colTree.sons_.size(); ++j) {
      assemblePattern(rowTree, colTree.sons_[j], triplet_list);
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<value_type>> triplet_list_;
  value_type a_param_;
  value_type dprime_;
  value_type dtilde_;
  value_type J_param_;
  value_type op_;
  IndexType max_wlevel_;
  value_type cut_const1_;
  value_type cut_const2_;
}; // namespace FMCA
} // namespace FMCA
#endif
