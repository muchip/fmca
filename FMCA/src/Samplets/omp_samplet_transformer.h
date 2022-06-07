// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_SAMPLETS_OMPSAMPLETTRANSFORMER_H_
#define FMCA_SAMPLETS_OMPSAMPLETTRANSFORMER_H_

namespace FMCA {

template <typename Derived> class ompSampletTransformer {
public:
  ompSampletTransformer() {}
  ompSampletTransformer(const SampletTreeBase<Derived> &ST) { init(ST); }

  void init(const SampletTreeBase<Derived> &ST) {
    max_level_ = 0;
    n_blocks_ = 0;
    // first sweep to determine samplet tree characteristics
    for (auto it = ST.cbegin(); it != ST.cend(); ++it) {
      const Derived &cluster = it->derived();
      max_level_ = max_level_ < cluster.level() ? cluster.level() : max_level_;
      ++n_blocks_;
    }
    tvec_.resize(n_blocks_);
    lvl_mapper_.resize(max_level_ + 1);
    // second sweep to map tree levels
    for (auto it = ST.cbegin(); it != ST.cend(); ++it) {
      const Derived &cluster = it->derived();
      lvl_mapper_[cluster.level()].push_back(std::addressof(cluster));
    }
    return;
  }
  template <typename otherDerived>
  Matrix transform(const Eigen::MatrixBase<otherDerived> &data) {
    // to parallelize, we need to avoid that a core accesses data that
    // has not been created yet to prevent this, we do a level wise blocking
    Matrix retval(data.rows(), data.cols());
    retval.setZero();
    for (auto it = lvl_mapper_.rbegin(); it != lvl_mapper_.rend(); ++it) {
#pragma omp for
      for (auto it2 = it->begin(); it2 != it->end(); ++it2) {
        const Derived &cluster = **it2;
        Matrix &block = tvec_[cluster.block_id()];
        if (!cluster.nSons())
          block = data.middleRows(cluster.indices_begin(),
                                  cluster.indices().size());
        else
          for (auto i = 0; i < cluster.nSons(); ++i) {
            block.conservativeResize(block.rows() + cluster.sons(i).nscalfs(),
                                     data.cols());
            block.bottomRows(cluster.sons(i).nscalfs()) =
                tvec_[cluster.sons(i).block_id()].topRows(
                    cluster.sons(i).nscalfs());
          }
        block = cluster.Q().transpose() * block;
        // write data to output
        if (cluster.is_root())
          retval.middleRows(cluster.start_index(), cluster.Q().cols()) =
              block.topRows(cluster.Q().cols());
        else if (cluster.nsamplets())
          retval.middleRows(cluster.start_index(), cluster.nsamplets()) =
              block.bottomRows(cluster.nsamplets());
      }
    }

    return retval;
  }

private:
  std::vector<Matrix> tvec_;
  std::vector<std::vector<const Derived *>> lvl_mapper_;
  Index n_blocks_;
  Index max_level_;
};
} // namespace FMCA

#endif
