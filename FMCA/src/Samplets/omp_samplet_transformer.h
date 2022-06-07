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
    Index max_level = 0;
    Index n_blocks = 0;
    // first sweep to determine samplet tree characteristics
    for (auto it = ST.cbegin(); it != ST.cend(); ++it) {
      const Derived &cluster = it->derived();
      max_level = max_level < cluster.level() ? cluster.level() : max_level;
      ++n_blocks;
    }
    tvec_.resize(n_blocks);
    lvl_mapper_.resize(max_level + 1);
    // second sweep to map tree levels
    for (auto it = ST.cbegin(); it != ST.cend(); ++it) {
      const Derived &cluster = it->derived();
      lvl_mapper_[cluster.level()].push_back(std::addressof(cluster));
    }
    return;
  }
  template <typename otherDerived>
  void transform(const Eigen::MatrixBase<otherDerived> &data) {
    // to parallelize, we need to avoid that a core accesses data that
    // has not been created yet to prevent this, we do a level wise blocking
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
        block = cluster.Q() * block;
      }
    }

    return;
  }

private:
  std::vector<Matrix> tvec_;
  std::vector<std::vector<const Derived *>> lvl_mapper_;
};
} // namespace FMCA

#endif
