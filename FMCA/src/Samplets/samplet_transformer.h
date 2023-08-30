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
#ifndef FMCA_SAMPLETS_SAMPLETTRANSFORMER_H_
#define FMCA_SAMPLETS_SAMPLETTRANSFORMER_H_

namespace FMCA {
namespace internal {
/**
 *  \ingroup Samplets
 *  \brief class which performs a samplet transform given a samplet tree
 *         iteratively with omp parallelism
 **/
template <typename Derived>
class SampletTransformer {
 public:
  SampletTransformer() {}
  SampletTransformer(const SampletTreeBase<Derived> &ST, Index min_level = 0) {
    init(ST, min_level);
  }

  void init(const SampletTreeBase<Derived> &ST, Index min_level = 0) {
    min_level_ = min_level;
    max_level_ = 0;
    n_blocks_ = 0;
    // first sweep to determine samplet tree characteristics
    for (const auto &cluster : ST) {
      max_level_ = max_level_ < cluster.level() ? cluster.level() : max_level_;
      ++n_blocks_;
    }
    tvec_.resize(n_blocks_);
    start_index_.resize(n_blocks_);
    lvl_mapper_.resize(max_level_ + 1 - min_level_);
    // second sweep to map tree levels
    Index start = 0;
    for (const auto &cluster : ST) {
      if (cluster.level() >= min_level_) {
        lvl_mapper_[cluster.level() - min_level_].push_back(
            std::addressof(cluster));
        start_index_[cluster.block_id()] = start;
        start += (cluster.level() == min_level_ ? cluster.Q().cols()
                                                : cluster.nsamplets());
      }
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
#pragma omp parallel for
      for (auto i = 0; i < it->size(); ++i) {
        const Derived &cluster = *((*it)[i]);
        Matrix &block = tvec_[cluster.block_id()];
        if (!cluster.nSons())
          block =
              data.middleRows(cluster.indices_begin(), cluster.block_size());
        else
          for (auto i = 0; i < cluster.nSons(); ++i) {
            block.conservativeResize(block.rows() + cluster.sons(i).nscalfs(),
                                     data.cols());
            block.bottomRows(cluster.sons(i).nscalfs()) =
                tvec_[cluster.sons(i).block_id()].topRows(
                    cluster.sons(i).nscalfs());
          }
        if (min_level_ > 0 && cluster.level() == min_level_)
          ;
        else
          block = cluster.Q().transpose() * block;
        // write data to output
        if (cluster.level() == min_level_)
          retval.middleRows(start_index_[cluster.block_id()],
                            cluster.Q().cols()) =
              block.topRows(cluster.Q().cols());
        else if (cluster.nsamplets())
          retval.middleRows(start_index_[cluster.block_id()],
                            cluster.nsamplets()) =
              block.bottomRows(cluster.nsamplets());
      }
    }

    return retval;
  }

  template <typename otherDerived>
  Matrix inverseTransform(const Eigen::MatrixBase<otherDerived> &data) {
    // to parallelize, we need to avoid that a core accesses data that
    // has not been created yet to prevent this, we do a level wise blocking
    Matrix retval(data.rows(), data.cols());
    retval.setZero();
    for (auto it = lvl_mapper_.begin(); it != lvl_mapper_.end(); ++it) {
#pragma omp parallel for
      for (auto i = 0; i < it->size(); ++i) {
        const Derived &cluster = *((*it)[i]);
        Matrix &block = tvec_[cluster.block_id()];
        if (cluster.level() == min_level_)
          block = data.middleRows(start_index_[cluster.block_id()],
                                  cluster.Q().cols());
        else {
          // since we have here the parallel version, we need to find
          // the chuck of scaling functions belonging to the current cluster
          // in the scaling functions of the dad
          Index data_offset = 0;
          for (auto j = 0; j < cluster.dad().nSons(); ++j)
            if (cluster.dad().sons(j).block_id() != cluster.block_id())
              data_offset += cluster.dad().sons(j).nscalfs();
            else
              break;
          block.topRows(cluster.nscalfs()) =
              tvec_[cluster.dad().block_id()].middleRows(data_offset,
                                                         cluster.nscalfs());
          block.bottomRows(cluster.nsamplets()) = data.middleRows(
              start_index_[cluster.block_id()], cluster.nsamplets());
        }
        if (min_level_ > 0 && cluster.level() == min_level_)
          ;
        else
          block = cluster.Q() * block;
        // write data to output
        if (!cluster.nSons())
          retval.middleRows(cluster.indices_begin(), block.rows()) = block;
      }
    }

    return retval;
  }

 private:
  std::vector<Matrix> tvec_;
  std::vector<std::vector<const Derived *>> lvl_mapper_;
  std::vector<Index> start_index_;
  Index n_blocks_;
  Index min_level_;
  Index max_level_;
};
}  // namespace internal
}  // namespace FMCA

#endif
