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
#include <queue>
#include <unordered_set>

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
    // Identificazione
    const Derived* row_cluster;
    const Derived* col_cluster;
    size_t block_key;  // row_id + nclusters * col_id
    Index level;

    // Dipendenze
    std::vector<MatrixBlockTask*> row_children;  // stessa colonna, righe figlie
    std::vector<MatrixBlockTask*> col_children;  // stessa riga, colonne figlie
    MatrixBlockTask* parent1 = nullptr;
    MatrixBlockTask* parent2 = nullptr;

    // Contatori
    int total_row_children = 0;
    int total_col_children = 0;
    int total_parents = 0;
    std::atomic<int> row_children_completed{0};
    std::atomic<int> col_children_completed{0};
    std::atomic<int> parents_notified{0};

    // Dati e stato
    Matrix block;
    std::atomic<bool> is_computed{false};
    std::atomic<bool> is_ready_notified{false};
    std::atomic<bool> is_queued{false};
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
      cv_.wait(lock, [this] {
        return !queue_.empty() || completed_tasks_ == total_tasks_;
      });

      if (queue_.empty()) return nullptr;

      MatrixBlockTask* task = queue_.front();
      queue_.pop();
      return task;
    }

    void mark_completed() {
      int done = ++completed_tasks_;
      if (done == total_tasks_) {
        cv_.notify_all();
      }
    }
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

  //////////////////////////////////////////////////////////////////////////////
  // INITIALIZATION
  //////////////////////////////////////////////////////////////////////////////

  void init(const SampletTreeBase<Derived>& ST, Scalar eta,
            Scalar threshold = 0) {
    eta_ = eta;
    threshold_ = threshold;
    npts_ = ST.block_size();
    rta_.init(ST, ST.block_size());

    const size_t nclusters = rta_.nodes().size();
    const int maxlevel = 2 * rta_.max_level() + 1;

    // FASE 1: BLOCK DISCOVERY
    // Per ogni coppia (row_cluster, col_cluster) ammissibile, trova il blocco
    
    std::vector<std::vector<std::unordered_set<size_t>>> thread_local_patterns;

#pragma omp parallel
    {
      const int num_threads = omp_get_num_threads();
      const int thread_id = omp_get_thread_num();

#pragma omp single
      {
        thread_local_patterns.resize(num_threads);
        for (auto& patterns : thread_local_patterns) {
          patterns.resize(maxlevel);
        }
      }

#pragma omp for schedule(dynamic)
      for (Index j = 0; j < nclusters; ++j) {
        const Derived* pc = rta_.nodes()[j];
        std::vector<const Derived*> row_stack{std::addressof(ST.derived())};

        while (!row_stack.empty()) {
          const Derived* pr = row_stack.back();
          row_stack.pop_back();

          for (auto i = 0; i < pr->nSons(); ++i) {
            if (ClusterComparison::compare(pr->sons(i), *pc, eta_) != LowRank) {
              row_stack.push_back(std::addressof(pr->sons(i)));
            }
          }

          if (pc->block_id() >= pr->block_id()) {
            const size_t block_key = pr->block_id() + nclusters * pc->block_id();
            const int level = pc->level() + pr->level();
            thread_local_patterns[thread_id][level].insert(block_key);
          }
        }
      }
    }

    // Merge
    std::vector<std::unordered_set<size_t>> level_patterns(maxlevel);
    for (const auto& thread_patterns : thread_local_patterns) {
      for (int level = 0; level < maxlevel; ++level) {
        level_patterns[level].insert(thread_patterns[level].begin(),
                                      thread_patterns[level].end());
      }
    }

    // FASE 2: TASK CREATION
    
    size_t total_tasks = 0;
    for (const auto& level_set : level_patterns) {
      total_tasks += level_set.size();
    }
    task_graph_.reserve(total_tasks);

    std::vector<std::vector<std::pair<size_t, std::unique_ptr<MatrixBlockTask>>>> 
        thread_local_tasks;

#pragma omp parallel
    {
      const int num_threads = omp_get_num_threads();
      const int thread_id = omp_get_thread_num();

#pragma omp single
      {
        thread_local_tasks.resize(num_threads);
      }

#pragma omp for schedule(dynamic)
      for (int level = 0; level < maxlevel; ++level) {
        for (size_t block_key : level_patterns[level]) {
          auto task = std::make_unique<MatrixBlockTask>();
          
          task->row_cluster = rta_.nodes()[block_key % nclusters];
          task->col_cluster = rta_.nodes()[block_key / nclusters];
          task->block_key = block_key;
          task->level = level;
          task->block = Matrix(0, 0);

          thread_local_tasks[thread_id].emplace_back(block_key, std::move(task));
        }
      }
    }

    for (auto& local_tasks : thread_local_tasks) {
      for (auto& [key, task] : local_tasks) {
        task_graph_[key] = std::move(task);
      }
    }

    // FASE 3: DEPENDENCY WIRING
    
    std::vector<std::pair<size_t, MatrixBlockTask*>> task_vec;
    task_vec.reserve(task_graph_.size());
    for (auto& [key, task] : task_graph_) {
      task_vec.emplace_back(key, task.get());
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t idx = 0; idx < task_vec.size(); ++idx) {
      auto* task = task_vec[idx].second;
      const Derived* pr = task->row_cluster;
      const Derived* pc = task->col_cluster;

      // Row children
      for (auto i = 0; i < pr->nSons(); ++i) {
        size_t child_key = pr->sons(i).block_id() + nclusters * pc->block_id();

        if (auto it = task_graph_.find(child_key); it != task_graph_.end()) {
          MatrixBlockTask* child = it->second.get();
          task->row_children.push_back(child);

#pragma omp critical
          {
            if (!child->parent1) {
              child->parent1 = task;
            } else if (!child->parent2) {
              child->parent2 = task;
            }
            child->total_parents++;
          }
        }
      }

      // Column children
      for (auto j = 0; j < pc->nSons(); ++j) {
        size_t child_key = pr->block_id() + nclusters * pc->sons(j).block_id();

        if (auto it = task_graph_.find(child_key); it != task_graph_.end()) {
          MatrixBlockTask* child = it->second.get();
          task->col_children.push_back(child);

#pragma omp critical
          {
            if (!child->parent1) {
              child->parent1 = task;
            } else if (!child->parent2) {
              child->parent2 = task;
            }
            child->total_parents++;
          }
        }
      }

      task->total_row_children = task->row_children.size();
      task->total_col_children = task->col_children.size();
    }

    // FASE 4: IDENTIFY LEAF TASKS
    
    leaf_tasks_.clear();
    leaf_tasks_.reserve(task_graph_.size() / 4);
    
    for (auto& [key, task] : task_graph_) {
      if (task->total_row_children == 0 && task->total_col_children == 0) {
        leaf_tasks_.push_back(task.get());
      }
    }

    task_queue_.set_total_tasks(task_graph_.size());

    std::cout << "Task graph initialized: " << task_graph_.size()
              << " total tasks, " << leaf_tasks_.size() << " leaf tasks"
              << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  // COMPRESSION
  //////////////////////////////////////////////////////////////////////////////

  template <typename EntGenerator>
  void compress(const EntGenerator& e_gen) {
    const auto nclusters = rta_.nodes().size();

    for (auto* leaf : leaf_tasks_) {
      task_queue_.enqueue(leaf);
    }

#pragma omp parallel
    {
      while (true) {
        MatrixBlockTask* task = task_queue_.dequeue();
        if (!task) break;

        computeBlock(task, e_gen, nclusters);
        notifyParents(task);
        task_queue_.mark_completed();
      }
    }
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

  template <typename EntGenerator>
  void computeBlock(MatrixBlockTask* task, const EntGenerator& e_gen,
                    size_t nclusters) {
    const Derived* pr = task->row_cluster;
    const Derived* pc = task->col_cluster;

    const char the_case = 2 * (!pr->nSons()) + (!pc->nSons());

    switch (the_case) {
      case 3: {  // (leaf, leaf)
        task->block = recursivelyComputeBlock(*pr, *pc, e_gen);
        break;
      }

      case 1: {  // (non-leaf, leaf)
        task->block.resize(pr->Q().rows(), pc->Q().cols());
        Index offset = 0;

        for (int k = 0; k < pr->nSons(); ++k) {
          const Index nscalfs = pr->sons(k).nscalfs();
          const size_t child_key =
              pr->sons(k).block_id() + nclusters * pc->block_id();

          if (auto it = task_graph_.find(child_key); it != task_graph_.end()) {
            task->block.middleRows(offset, nscalfs) =
                it->second->block.topRows(nscalfs);
          } else {
            const Matrix ret = recursivelyComputeBlock(pr->sons(k), *pc, e_gen);
            task->block.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
          }
          offset += nscalfs;
        }

        task->block = pr->Q().transpose() * task->block;
        break;
      }

      case 2:
      case 0: {  // (leaf, non-leaf) o (non-leaf, non-leaf)
        const bool row_ready =
            (task->row_children_completed.load() == task->total_row_children);
        const bool col_ready =
            (task->col_children_completed.load() == task->total_col_children);

        bool use_row_children = false;

        if (row_ready && col_ready) {
          use_row_children =
              (task->total_row_children <= task->total_col_children);
        } else if (row_ready) {
          use_row_children = true;
        } else if (col_ready) {
          use_row_children = false;
        }

        if (use_row_children && row_ready) {
          task->block.resize(pr->Q().rows(), pc->Q().cols());
          Index offset = 0;

          for (int k = 0; k < pr->nSons(); ++k) {
            const Index nscalfs = pr->sons(k).nscalfs();
            const size_t child_key =
                pr->sons(k).block_id() + nclusters * pc->block_id();

            if (auto it = task_graph_.find(child_key);
                it != task_graph_.end()) {
              task->block.middleRows(offset, nscalfs) =
                  it->second->block.topRows(nscalfs);
            } else {
              const Matrix ret =
                  recursivelyComputeBlock(pr->sons(k), *pc, e_gen);
              task->block.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
            }
            offset += nscalfs;
          }

          task->block = pr->Q().transpose() * task->block;

        } else if (!use_row_children && col_ready) {
          task->block.resize(pr->Q().cols(), pc->Q().rows());
          Index offset = 0;

          for (int k = 0; k < pc->nSons(); ++k) {
            const Index nscalfs = pc->sons(k).nscalfs();
            const size_t child_key =
                pc->sons(k).block_id() * nclusters + pr->block_id();

            if (auto it = task_graph_.find(child_key);
                it != task_graph_.end()) {
              task->block.middleCols(offset, nscalfs) =
                  it->second->block.leftCols(nscalfs);
            } else {
              const Matrix ret =
                  recursivelyComputeBlock(*pr, pc->sons(k), e_gen);
              task->block.middleCols(offset, nscalfs) = ret.leftCols(nscalfs);
            }
            offset += nscalfs;
          }

          task->block = task->block * pc->Q();
        } else {
          task->block = recursivelyComputeBlock(*pr, *pc, e_gen);
        }

        break;
      }
    }

    task->is_computed.store(true);
  }

  //////////////////////////////////////////////////////////////////////////////
  // PARENT NOTIFICATION
  //////////////////////////////////////////////////////////////////////////////

  void notifyParents(MatrixBlockTask* task) {
    if (task->parent1) {
      updateParent(task, task->parent1);
    }
    if (task->parent2) {
      updateParent(task, task->parent2);
    }
  }

  void updateParent(MatrixBlockTask* child, MatrixBlockTask* parent) {
    // Calcola al volo se child è row o col child del parent
    bool is_row_child = false;
    bool is_col_child = false;

    for (auto* rc : parent->row_children) {
      if (rc == child) {
        is_row_child = true;
        break;
      }
    }

    for (auto* cc : parent->col_children) {
      if (cc == child) {
        is_col_child = true;
        break;
      }
    }

    // Aggiorna contatori
    if (is_row_child) {
      parent->row_children_completed.fetch_add(1);
    }
    if (is_col_child) {
      parent->col_children_completed.fetch_add(1);
    }

    // Controlla se parent è pronto
    if (!parent->is_ready_notified.exchange(true)) {
      const int row_completed = parent->row_children_completed.load();
      const int col_completed = parent->col_children_completed.load();

      const Derived* pr = parent->row_cluster;
      const Derived* pc = parent->col_cluster;
      const char the_case = 2 * (!pr->nSons()) + (!pc->nSons());

      bool parent_ready = false;

      switch (the_case) {
        case 3:
          parent_ready = true;
          break;
        case 1:
          parent_ready = (row_completed == parent->total_row_children);
          break;
        case 2:
          parent_ready = (col_completed == parent->total_col_children);
          break;
        case 0:
          parent_ready = (row_completed == parent->total_row_children) ||
                         (col_completed == parent->total_col_children);
          break;
      }

      if (parent_ready) {
        task_queue_.enqueue(parent);
      } else {
        parent->is_ready_notified.store(false);
      }
    }

    child->parents_notified.fetch_add(1);
  }

  //////////////////////////////////////////////////////////////////////////////
  // RECURSIVE COMPUTATION
  //////////////////////////////////////////////////////////////////////////////

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
  std::vector<MatrixBlockTask*> leaf_tasks_;
  TaskQueue task_queue_;
};

}  // namespace internal
}  // namespace FMCA

#endif