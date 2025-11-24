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
#ifndef FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSOR_H_
#define FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSOR_H_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include "../util/RandomTreeAccessor.h"

namespace FMCA {
namespace internal {

template <typename Derived, typename ClusterComparison = CompareCluster>
class SampletMatrixCompressor {
 public:
  //////////////////////////////////////////////////////////////////////////////
  // TASK DEFINITION
  //////////////////////////////////////////////////////////////////////////////

  struct MatrixBlockTask {
    // Identification
    const Derived* row_cluster;  // τ
    const Derived* col_cluster;  // τ'
    size_t block_key;
    Index level;

    // Data
    Matrix block;

    // Dependencies
    std::vector<MatrixBlockTask*> row_children;  // (τ_child, τ')
    std::vector<MatrixBlockTask*> col_children;  // (τ, τ'_child)
    MatrixBlockTask* row_parent = nullptr;       // (parent(τ), τ')
    MatrixBlockTask* col_parent = nullptr;       // (τ, parent(τ'))

    // Atomic counters for synchronization
    std::atomic<int> row_children_completed{0};
    std::atomic<int> col_children_completed{0};

    // State flags
    std::atomic<bool> is_computed{false};
    std::atomic<bool> is_queued{false};

    MatrixBlockTask(const Derived* pr, const Derived* pc, size_t key)
        : row_cluster(pr),
          col_cluster(pc),
          block_key(key),
          level(pr->level() + pc->level()),
          block(0, 0) {}

    int total_row_children() const { return row_children.size(); }
    int total_col_children() const { return col_children.size(); }
  };

  //////////////////////////////////////////////////////////////////////////////
  // TASK QUEUE
  //////////////////////////////////////////////////////////////////////////////

  class TaskQueue {
   private:
    std::queue<MatrixBlockTask*> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<int> completed_tasks_{0};
    int total_tasks_;
    std::atomic<bool> done_{false};

   public:
    void set_total_tasks(int n) { total_tasks_ = n; }

    void enqueue(MatrixBlockTask* task) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!task->is_queued.exchange(true)) {
        queue_.push(task);
        cv_.notify_one();
      }
    }

    MatrixBlockTask* dequeue() {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return !queue_.empty() || done_.load(); });

      if (queue_.empty()) return nullptr;

      MatrixBlockTask* task = queue_.front();
      queue_.pop();
      return task;
    }

    void mark_completed() {
      int done = ++completed_tasks_;
      if (done == total_tasks_) {
        done_.store(true);
        cv_.notify_all();
      }
    }

    bool is_done() const { return done_.load(); }
  };

  //////////////////////////////////////////////////////////////////////////////
  // CONSTRUCTOR & ACCESSORS
  //////////////////////////////////////////////////////////////////////////////

  SampletMatrixCompressor() {}

  SampletMatrixCompressor(const SampletTreeBase<Derived>& ST, Scalar eta,
                          Scalar threshold = 0) {
    init(ST, eta, threshold);
  }

  const RandomTreeAccessor<Derived>& rta() const { return rta_; }

  const std::unordered_map<size_t, std::unique_ptr<MatrixBlockTask>>&
  task_graph() const {
    return task_graph_;
  }

  //////////////////////////////////////////////////////////////////////////////
  // INITIALIZATION
  //////////////////////////////////////////////////////////////////////////////

  /**
   * \brief Creates the task graph with admissible blocks and dependencies
   */
  void init(const SampletTreeBase<Derived>& ST, Scalar eta,
            Scalar threshold = 0) {
    eta_ = eta;
    threshold_ = threshold;
    npts_ = ST.block_size();
    rta_.init(ST, ST.block_size());

    const size_t n_nodes = rta_.nodes().size();

    // Phase 1: Build task graph with admissible blocks
    std::mutex graph_mutex;

#pragma omp parallel for schedule(dynamic)
    for (Index j = 0; j < n_nodes; ++j) {
      const Derived* pc = rta_.nodes()[j];

      // DFS on rows for this column
      std::vector<const Derived*> row_stack;
      row_stack.push_back(std::addressof(ST.derived()));

      // Local buffer for blocks found by this thread
      std::vector<std::tuple<size_t, const Derived*, const Derived*>>
          local_blocks;

      while (!row_stack.empty()) {
        const Derived* pr = row_stack.back();
        row_stack.pop_back();

        // Add admissible children to stack
        for (auto i = 0; i < pr->nSons(); ++i) {
          if (ClusterComparison::compare(pr->sons(i), *pc, eta) != LowRank) {
            row_stack.push_back(std::addressof(pr->sons(i)));
          }
        }

        // Store block if in upper half (symmetry)
        if (pc->block_id() >= pr->block_id()) {
          const size_t block_key = pr->block_id() + n_nodes * pc->block_id();
          local_blocks.emplace_back(block_key, pr, pc);
        }
      }

      // Batch insert with single lock per column
      if (!local_blocks.empty()) {
        std::lock_guard<std::mutex> lock(graph_mutex);
        for (const auto& [key, pr, pc] : local_blocks) {
          task_graph_.emplace(key,
                              std::make_unique<MatrixBlockTask>(pr, pc, key));
        }
      }
    }

    // Phase 2: Build dependencies (children + parents)
    for (auto& [key, task] : task_graph_) {
      const Derived* pr = task->row_cluster;
      const Derived* pc = task->col_cluster;

      // Row children: (τ_child, τ') - same column, finer rows
      if (pr->nSons() > 0) {
        for (int i = 0; i < pr->nSons(); ++i) {
          const size_t child_key =
              pr->sons(i).block_id() + n_nodes * pc->block_id();

          auto it = task_graph_.find(child_key);
          if (it != task_graph_.end()) {
            task->row_children.push_back(it->second.get());
            it->second->row_parent = task.get();
          }
        }
      }

      // Column children: (τ, τ'_child) - same row, finer columns
      if (pc->nSons() > 0) {
        for (int i = 0; i < pc->nSons(); ++i) {
          const size_t child_key =
              pr->block_id() + n_nodes * pc->sons(i).block_id();

          auto it = task_graph_.find(child_key);
          if (it != task_graph_.end()) {
            task->col_children.push_back(it->second.get());
            it->second->col_parent = task.get();
          }
        }
      }
    }

    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // COMPRESSION
  //////////////////////////////////////////////////////////////////////////////

  /**
   * \brief Compresses the matrix using DAG-based parallel execution
   */
  template <typename EntGenerator>
  void compress(const EntGenerator& e_gen) {
    const size_t nclusters = rta_.nodes().size();

    // Identify leaf tasks (no children, ready immediately)
    std::vector<MatrixBlockTask*> leaf_tasks;
    for (auto& [key, task] : task_graph_) {
      if (task->total_row_children() == 0 && task->total_col_children() == 0) {
        leaf_tasks.push_back(task.get());
      }
    }

    // Initialize task queue
    TaskQueue task_queue;
    task_queue.set_total_tasks(task_graph_.size());

    for (auto* leaf : leaf_tasks) {
      task_queue.enqueue(leaf);
    }

    // Parallel execution
#pragma omp parallel
    {
      while (true) {
        MatrixBlockTask* task = task_queue.dequeue();
        if (!task) break;

        compute_block(task, e_gen, nclusters);
        task->is_computed.store(true);
        task_queue.mark_completed();
        notify_parents(task, task_queue);
      }
    }

    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // TRIPLET GENERATION
  //////////////////////////////////////////////////////////////////////////////

  const std::vector<Triplet<Scalar>>& triplets() {
    triplet_list_.clear();

    std::vector<MatrixBlockTask*> all_tasks_vec;
    all_tasks_vec.reserve(task_graph_.size());
    for (auto& [key, task] : task_graph_) {
      all_tasks_vec.push_back(task.get());
    }

#pragma omp parallel
    {
      std::vector<Triplet<Scalar>> local_list;

#pragma omp for schedule(dynamic) nowait
      for (size_t i = 0; i < all_tasks_vec.size(); ++i) {
        const auto* task = all_tasks_vec[i];
        const Derived* pr = task->row_cluster;
        const Derived* pc = task->col_cluster;

        if (!pr->is_root() && !pc->is_root()) {
          storeBlock(
              local_list, pr->start_index(), pc->start_index(), pr->nsamplets(),
              pc->nsamplets(),
              task->block.bottomRightCorner(pr->nsamplets(), pc->nsamplets()));
        } else if (!pc->is_root()) {
          storeBlock(local_list, pr->start_index(), pc->start_index(),
                     pr->Q().cols(), pc->nsamplets(),
                     task->block.rightCols(pc->nsamplets()));
        } else if (pr->is_root() && pc->is_root()) {
          storeBlock(local_list, pr->start_index(), pc->start_index(),
                     pr->Q().cols(), pc->Q().cols(), task->block);
        }
      }

#pragma omp critical
      triplet_list_.insert(triplet_list_.end(), local_list.begin(),
                           local_list.end());
    }

    for (auto& [key, task] : task_graph_) {
      task->block.resize(0, 0);
    }

    return triplet_list_;
  }

  std::vector<Eigen::Triplet<Scalar>> a_priori_pattern_triplets() {
    std::vector<Eigen::Triplet<Scalar>> retval;

    std::vector<MatrixBlockTask*> all_tasks_vec;
    all_tasks_vec.reserve(task_graph_.size());
    for (auto& [key, task] : task_graph_) {
      all_tasks_vec.push_back(task.get());
    }

#pragma omp parallel
    {
      std::vector<Triplet<Scalar>> local_list;

#pragma omp for schedule(dynamic) nowait
      for (size_t i = 0; i < all_tasks_vec.size(); ++i) {
        const auto* task = all_tasks_vec[i];
        const Derived* pr = task->row_cluster;
        const Derived* pc = task->col_cluster;

        if (!pr->is_root() && !pc->is_root())
          storeEmptyBlock(local_list, pr->start_index(), pc->start_index(),
                          pr->nsamplets(), pc->nsamplets());
        else if (!pc->is_root())
          storeEmptyBlock(local_list, pr->start_index(), pc->start_index(),
                          pr->Q().cols(), pc->nsamplets());
        else if (pr->is_root() && pc->is_root())
          storeEmptyBlock(local_list, pr->start_index(), pc->start_index(),
                          pr->Q().cols(), pc->Q().cols());
      }

#pragma omp critical
      retval.insert(retval.end(), local_list.begin(), local_list.end());
    }

    return retval;
  }

  std::vector<Triplet<Scalar>> aposteriori_triplets_fast(const Scalar thres) {
    std::vector<Triplet<Scalar>> retval;
    std::vector<std::vector<Index>> buckets(17);
    std::vector<Scalar> norms2(17);
    const Scalar invlog10 = 1. / std::log(10.);

    for (FMCA::Index i = 0; i < triplet_list_.size(); ++i) {
      const Scalar entry = std::abs(triplet_list_[i].value());
      const Scalar val = -std::floor(invlog10 * std::log(entry));
      const Index ind = val < 0 ? 0 : val;
      buckets[ind > 16 ? 16 : ind].push_back(i);
      norms2[ind > 16 ? 16 : ind] += entry * entry;
    }

    Scalar fnorm2 = 0;
    for (int i = 16; i >= 0; --i) fnorm2 += norms2[i];

    Scalar cut_snorm = 0;
    Index cut_off = 17;
    for (int i = 16; i >= 0; --i) {
      cut_snorm += norms2[i];
      if (std::sqrt(cut_snorm / fnorm2) >= thres) break;
      --cut_off;
    }

    Index ntriplets = 0;
    for (Index i = 0; i < cut_off; ++i) ntriplets += buckets[i].size();
    retval.reserve(ntriplets + npts_);

    for (Index i = 0; i < cut_off; ++i)
      for (const auto& it : buckets[i]) retval.push_back(triplet_list_[it]);

    for (Index i = cut_off; i < 17; ++i)
      for (const auto& it : buckets[i])
        if (triplet_list_[it].row() == triplet_list_[it].col())
          retval.push_back(triplet_list_[it]);

    retval.shrink_to_fit();
    return retval;
  }

  std::vector<Triplet<Scalar>> aposteriori_triplets(const Scalar thres) {
    std::vector<Triplet<Scalar>> triplets = triplet_list_;
    if (std::abs(thres) < FMCA_ZERO_TOLERANCE) return triplets;

    std::vector<long int> idcs(triplet_list_.size());
    std::iota(idcs.begin(), idcs.end(), 0);

    {
      struct comp {
        comp(const std::vector<Triplet<Scalar>>& triplets) : ts_(triplets) {}
        bool operator()(const Index& a, const Index& b) const {
          const Scalar val1 = (ts_[a].row() == ts_[a].col())
                                  ? FMCA_INF
                                  : std::abs(ts_[a].value());
          const Scalar val2 = (ts_[b].row() == ts_[b].col())
                                  ? FMCA_INF
                                  : std::abs(ts_[b].value());
          return val1 > val2;
        }
        const std::vector<Triplet<Scalar>>& ts_;
      };
      std::sort(idcs.begin(), idcs.end(), comp(triplet_list_));
    }

    Scalar squared_norm = 0;
    for (auto it = idcs.rbegin(); it != idcs.rend(); ++it)
      squared_norm += triplet_list_[*it].value() * triplet_list_[*it].value();

    Scalar cut_snorm = 0;
    Index cut_off = triplet_list_.size();
    for (auto it = idcs.rbegin(); it != idcs.rend(); ++it) {
      cut_snorm += triplet_list_[*it].value() * triplet_list_[*it].value();
      if (std::sqrt(cut_snorm / squared_norm) >= thres) break;
      --cut_off;
    }

    cut_off = cut_off < npts_ ? npts_ : cut_off;
    idcs.resize(cut_off);
    triplets.resize(cut_off);
    for (Index i = 0; i < cut_off; ++i) triplets[i] = triplet_list_[idcs[i]];

    return triplets;
  }

  std::vector<Eigen::Triplet<Scalar>> release_triplets() {
    std::vector<Eigen::Triplet<Scalar>> retval;
    std::swap(triplet_list_, retval);
    return retval;
  }

 private:
  //////////////////////////////////////////////////////////////////////////////
  // BLOCK COMPUTATION
  //////////////////////////////////////////////////////////////////////////////

  /**
   * \brief Computes a single matrix block based on cluster tree structure
   *
   * Three cases:
   * 1. (LEAF, LEAF): Compute from scratch
   * 2. (NON-LEAF, LEAF): Recycle from row children
   * 3. (*, NON-LEAF): Recycle from column children
   */
  template <typename EntGenerator>
  void compute_block(MatrixBlockTask* task, const EntGenerator& e_gen,
                     size_t nclusters) {
    const Derived* pr = task->row_cluster;
    const Derived* pc = task->col_cluster;

    const bool row_is_leaf = (pr->nSons() == 0);
    const bool col_is_leaf = (pc->nSons() == 0);

    // Case 1: (LEAF, LEAF)
    if (row_is_leaf && col_is_leaf) {
      task->block = recursivelyComputeBlock(*pr, *pc, e_gen);
      return;
    }

    // Case 2: (NON-LEAF, LEAF) - recycle from row children
    if (!row_is_leaf && col_is_leaf) {
      task->block.resize(pr->Q().rows(), pc->Q().cols());

      Index offset = 0;
      for (int k = 0; k < pr->nSons(); ++k) {
        const Derived* pr_child = &pr->sons(k);
        const Index nscalfs = pr_child->nscalfs();

        const size_t child_key =
            pr_child->block_id() + nclusters * pc->block_id();
        auto it = task_graph_.find(child_key);

        if (it != task_graph_.end()) {
          // Reuse existing block
          task->block.middleRows(offset, nscalfs) =
              it->second->block.topRows(nscalfs);
        } else {
          // Compute on-the-fly (child not in task_graph due to symmetry)
          Matrix ret = recursivelyComputeBlock(*pr_child, *pc, e_gen);
          task->block.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
        }

        offset += nscalfs;
      }

      task->block = pr->Q().transpose() * task->block;
      return;
    }

    // Case 3: (*, NON-LEAF) - recycle from column children
    if (!col_is_leaf) {
      task->block.resize(pr->Q().cols(), pc->Q().rows());

      Index offset = 0;
      for (int k = 0; k < pc->nSons(); ++k) {
        const Derived* pc_child = &pc->sons(k);
        const Index nscalfs = pc_child->nscalfs();

        const size_t child_key =
            pr->block_id() + nclusters * pc_child->block_id();
        auto it = task_graph_.find(child_key);

        if (it != task_graph_.end()) {
          // Reuse existing block
          task->block.middleCols(offset, nscalfs) =
              it->second->block.leftCols(nscalfs);
        } else {
          // Compute on-the-fly (child not in task_graph due to symmetry)
          Matrix ret = recursivelyComputeBlock(*pr, *pc_child, e_gen);
          task->block.middleCols(offset, nscalfs) = ret.leftCols(nscalfs);
        }

        offset += nscalfs;
      }

      task->block = task->block * pc->Q();
      return;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // PARENT NOTIFICATION
  //////////////////////////////////////////////////////////////////////////////

  /**
   * \brief Notifies parents that a child has been computed
   */
  void notify_parents(MatrixBlockTask* completed_child,
                      TaskQueue& task_queue) {
    if (completed_child->row_parent) {
      notify_single_parent(completed_child, completed_child->row_parent, true,
                           task_queue);
    }

    if (completed_child->col_parent) {
      notify_single_parent(completed_child, completed_child->col_parent, false,
                           task_queue);
    }
  }

  /**
   * \brief Notifies a single parent and enqueues it if ready
   */
  void notify_single_parent(MatrixBlockTask* child, MatrixBlockTask* parent,
                            bool is_row_child, TaskQueue& task_queue) {
    // Update completion counters
    int row_done, col_done;

    if (is_row_child) {
      row_done = ++parent->row_children_completed;
      col_done = parent->col_children_completed.load();
    } else {
      col_done = ++parent->col_children_completed;
      row_done = parent->row_children_completed.load();
    }

    // Check if parent is ready to be computed
    const bool row_is_leaf = (parent->row_cluster->nSons() == 0);
    const bool col_is_leaf = (parent->col_cluster->nSons() == 0);

    bool is_ready = false;

    if (!row_is_leaf && col_is_leaf) {
      // Case 2: needs all row children
      is_ready = (row_done == parent->total_row_children());
    } else if (!col_is_leaf) {
      // Case 3: needs all column children (ignores row children if present)
      is_ready = (col_done == parent->total_col_children());
    }

    if (is_ready) {
      task_queue.enqueue(parent);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // RECURSIVE COMPUTATION
  //////////////////////////////////////////////////////////////////////////////

  /**
   * \brief Recursively computes a matrix block with interpolation and
   *        transformation
   */
  template <typename EntryGenerator>
  Matrix recursivelyComputeBlock(const Derived& TR, const Derived& TC,
                                 const EntryGenerator& e_gen) {
    Matrix buf(0, 0);

    if (ClusterComparison::compare(TR, TC, eta_) == LowRank) {
      e_gen.interpolate_kernel(TR, TC, &buf);
      return TR.V().transpose() * buf * TC.V();
    } else {
      const char the_case = 2 * (!TR.nSons()) + !TC.nSons();

      switch (the_case) {
        case 3:
          e_gen.compute_dense_block(TR, TC, &buf);
          return TR.Q().transpose() * buf * TC.Q();

        case 2:
          for (auto j = 0; j < TC.nSons(); ++j) {
            const Index nscalfs = TC.sons(j).nscalfs();
            Matrix ret = recursivelyComputeBlock(TR, TC.sons(j), e_gen);
            buf.conservativeResize(ret.rows(), buf.cols() + nscalfs);
            buf.rightCols(nscalfs) = ret.leftCols(nscalfs);
          }
          return buf * TC.Q();

        case 1:
          for (auto i = 0; i < TR.nSons(); ++i) {
            const Index nscalfs = TR.sons(i).nscalfs();
            Matrix ret = recursivelyComputeBlock(TR.sons(i), TC, e_gen);
            buf.conservativeResize(ret.cols(), buf.cols() + nscalfs);
            buf.rightCols(nscalfs) = ret.transpose().leftCols(nscalfs);
          }
          return (buf * TR.Q()).transpose();

        case 0:
          for (auto i = 0; i < TR.nSons(); ++i) {
            Matrix ret1(0, 0);
            const Index r_nscalfs = TR.sons(i).nscalfs();
            for (auto j = 0; j < TC.nSons(); ++j) {
              const Index c_nscalfs = TC.sons(j).nscalfs();
              Matrix ret2 =
                  recursivelyComputeBlock(TR.sons(i), TC.sons(j), e_gen);
              ret1.conservativeResize(ret2.rows(), ret1.cols() + c_nscalfs);
              ret1.rightCols(c_nscalfs) = ret2.leftCols(c_nscalfs);
            }
            ret1 = ret1 * TC.Q();
            buf.conservativeResize(ret1.cols(), buf.cols() + r_nscalfs);
            buf.rightCols(r_nscalfs) = ret1.transpose().leftCols(r_nscalfs);
          }
          return (buf * TR.Q()).transpose();
      }
    }
    return Matrix(0, 0);
  }

  //////////////////////////////////////////////////////////////////////////////
  // TRIPLET STORAGE
  //////////////////////////////////////////////////////////////////////////////

  void storeBlock(std::vector<Eigen::Triplet<Scalar>>& triplet_buffer,
                  Index srow, Index scol, Index nrows, Index ncols,
                  const Matrix& block) {
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if ((srow + j <= scol + k && std::abs(block(j, k)) > threshold_) ||
            (srow == scol && j == k))
          triplet_buffer.push_back(
              Eigen::Triplet<Scalar>(srow + j, scol + k, block(j, k)));
  }

  void storeEmptyBlock(std::vector<Eigen::Triplet<Scalar>>& triplet_buffer,
                       Index srow, Index scol, Index nrows, Index ncols) {
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if (srow + j <= scol + k)
          triplet_buffer.push_back(
              Eigen::Triplet<Scalar>(srow + j, scol + k, 0));
  }

  //////////////////////////////////////////////////////////////////////////////
  // MEMBER VARIABLES
  //////////////////////////////////////////////////////////////////////////////

  std::vector<Triplet<Scalar>> triplet_list_;
  RandomTreeAccessor<Derived> rta_;
  Scalar eta_;
  Scalar threshold_;
  Index npts_;

  std::unordered_map<size_t, std::unique_ptr<MatrixBlockTask>> task_graph_;
};

}  // namespace internal
}  // namespace FMCA

#endif