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
  ////////////////////////////////////////////////////////////////////////////////////
  // Represents a block-task in the dependency graph
  struct MatrixBlockTask {
    const Derived* row_cluster;
    const Derived* col_cluster;
    size_t block_key;  // row_id + nclusters * col_id
    Index level;       // row level + col level

    // Dependency info
    std::vector<MatrixBlockTask*> children;  // blocks we depend on
    MatrixBlockTask* parent1 = nullptr;
    MatrixBlockTask* parent2 = nullptr;
    int total_children = 0;
    int total_parents = 0;

    Matrix block;  // block matrix that we compiute

    std::atomic<int> children_completed{
        0};  // children that have finished computing
    std::atomic<int> parents_notified{
        0};  // parents that have notified us they're done using our block

    std::atomic<bool> is_computed{false};     // have we computed our block
    std::atomic<bool> is_queued{false};       // are we in the ready queue
    std::atomic<bool> memory_reduced{false};  // have we done garbage collection
  };

  ////////////////////////////////////////////////////////////////////////////////////
  class TaskQueue {
   private:
    std::queue<MatrixBlockTask*> queue_;  // pointers to ready tasks, FIFO
    std::mutex mutex_;  // mutual exclusion: only one thread can access queue_
                        // at a time
    std::condition_variable
        cv_;  // threads sleep until they are notified. cv.wait() = goes to
              // sleep; cv_.notify_one() = wakes up ONE sleeping thread

    std::atomic<int> completed_tasks_{
        0};  // atomic = multiple threads increment this simultaneously
    int total_tasks_;

   public:
    void set_total_tasks(int n) { total_tasks_ = n; }

    //---------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------
    // Add a task to the queue (thread-safe)
    // Uses exchange to ensure we don't queue the same task twice
    void enqueue(MatrixBlockTask* task) {
      std::lock_guard<std::mutex> lock(mutex_);
      // exchange(true): atomically does:
      //   old_value = is_queued;  // read old value
      //   is_queued = true;        // set to true
      //   return old_value;        // return what it was before
      if (!task->is_queued.exchange(true)) {
        queue_.push(task);
        cv_.notify_one();  // wake up one waiting thread
      }
    }

    // Get a task from the queue (blocks until task available or all done)
    MatrixBlockTask* dequeue() {
      std::unique_lock<std::mutex> lock(mutex_);
      // Wait until: queue has something OR all tasks are done
      cv_.wait(lock, [this] {
        return !queue_.empty() || completed_tasks_ == total_tasks_;
      });

      // If queue is empty, it means all tasks are done
      if (queue_.empty()) return nullptr;

      MatrixBlockTask* task = queue_.front();
      queue_.pop();
      return task;
    }

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
    // Mark a task as completed
    void mark_completed() {
      int done = ++completed_tasks_;
      // If we just completed the last task, wake up ALL threads so they can
      // exit
      if (done == total_tasks_) {
        cv_.notify_all();
      }
    }
  };

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
  SampletMatrixCompressor() {}
  SampletMatrixCompressor(const SampletTreeBase<Derived>& ST, Scalar eta,
                          Scalar threshold = 0) {
    init(ST, eta, threshold);
  }
  const RandomTreeAccessor<Derived>& rta() { return rta_; };
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
    void init(const SampletTreeBase<Derived>& ST, Scalar eta,
              Scalar threshold = 0) {
      eta_ = eta;
      threshold_ = threshold;
      npts_ = ST.block_size();
      rta_.init(ST, ST.block_size());

      const size_t nclusters = rta_.nodes().size();
      const int maxlevel = 2 * rta_.max_level() + 1;

      // 1: BLOCK DISCOVERY
      // We perform a DFS for each column cluster to find all admissible blocks. This creates the initial set of tasks we need to compute.
      // Temporary storage: organize blocks by level
      std::vector<std::unordered_set<size_t>> level_patterns(maxlevel);  // FIND A BETTER SOLUTION!!!!!!!!!!!!!

  #pragma omp parallel for schedule(dynamic)
      for (Index j = 0; j < nclusters; ++j) {
        const Derived* pc = rta_.nodes()[j];
        // DFS stack for row clusters
        std::vector<const Derived*> row_stack{std::addressof(ST.derived())};

        while (!row_stack.empty()) {
          const Derived* pr = row_stack.back();
          row_stack.pop_back();

          // Push children to stack if they're not admissible (i.e., we need to go deeper)
          for (auto i = 0; i < pr->nSons(); ++i) {
            if (ClusterComparison::compare(pr->sons(i), *pc, eta_) !=
            LowRank) {
              row_stack.push_back(std::addressof(pr->sons(i)));
            }
          }

          // Only store blocks in upper triangular part (col_id >= row_id)
          if (pc->block_id() >= pr->block_id()) {
            const size_t block_key = pr->block_id() + nclusters *
            pc->block_id(); const int level = pc->level() + pr->level();

  #pragma omp critical
            {
              level_patterns[level].insert(block_key);
            }
          }
        }
      }

      // 2: TASK CREATION
      // Now create the actual MatrixBlockTask objects for all discovered blocks

      task_graph_.clear();

      for (int level = 0; level < maxlevel; ++level) {
        for (size_t block_key : level_patterns[level]) {
          auto task = std::make_unique<MatrixBlockTask>(); // Create a new task using unique_ptr

          // Decode the key back to row and column IDs
          task->row_cluster = rta_.nodes()[block_key % nclusters];
          task->col_cluster = rta_.nodes()[block_key / nclusters];
          task->block_key = block_key;
          task->level = level;

          // Initialize the block as empty
          task->block = Matrix(0, 0);

          // Store in the graph
          task_graph_[block_key] = std::move(task); 
          // std::move transfers ownership of unique_ptr to the map
          // After this line, 'task' is empty (nullptr)
          // The MatrixBlockTask object now lives in task_graph_[block_key]
        }
      }

      // 3: DEPENDENCY WIRING
      // For each task, identify its children (dependencies) and wire up the graph. A task depends on children if it needs to recycle their computed blocks.

      for (auto& [key, task] : task_graph_) {
        const Derived* pr = task->row_cluster;
        const Derived* pc = task->col_cluster;
        // --- ROW CHILDREN ---
        // If the row cluster has sons, this task might depend on blocks
        // with the same column but child row clusters (case 1: noleaf, leaf)
        for (auto i = 0; i < pr->nSons(); ++i) {
          size_t child_key = pr->sons(i).block_id() + nclusters *
          pc->block_id();
              // Compute the key for the potential child block
              // This child has:
              //   - Row: pr->sons(i) (child of current row)
              //   - Column: pc (same column as parent)

          if (auto it = task_graph_.find(child_key); it != task_graph_.end()) // Check if this child block exists in our task graph
          {
            MatrixBlockTask* child = it->second.get();

            // Add to our children list
            // This task now depends on 'child'
            // We must wait for 'child' to finish before we can compute
            task->children.push_back(child);

            // Set ourselves as a parent of the child
            if (!child->parent1)
              child->parent1 = task.get();
            else if (!child->parent2)
              child->parent2 = task.get();

            child->total_parents++;
            // Child needs to know: "how many parents depend on me?"
            // For garbage collection: child can free memory only after
            // ALL parents have finished using its data
          }
        }

        // --- COLUMN CHILDREN ---
        // If the column cluster has sons, this task might depend on blocks
        // with the same row but child column clusters (case 0, 2)
        for (auto j = 0; j < pc->nSons(); ++j) {
          size_t child_key = pr->block_id() + nclusters *
          pc->sons(j).block_id();

          if (auto it = task_graph_.find(child_key); it != task_graph_.end())
          {
            MatrixBlockTask* child = it->second.get();

            // Add to our children list
            task->children.push_back(child);

            // Set ourselves as a parent of the child
            if (!child->parent1)
              child->parent1 = task.get();
            else if (!child->parent2)
              child->parent2 = task.get();

            child->total_parents++;
          }
        }

        task->total_children = task->children.size();
      }

      // 4: IDENTIFY LEAF TASKS
      // Leaf tasks are those with no children - they can be computed immediately without waiting for any dependencies
      leaf_tasks_.clear();
      for (auto& [key, task] : task_graph_) {
        if (task->total_children == 0) {
          leaf_tasks_.push_back(task.get());
        }
      }
      // Set up the task queue
      task_queue_.set_total_tasks(task_graph_.size());

      std::cout << "Task graph initialized: " << task_graph_.size()
                << " total tasks, " << leaf_tasks_.size() << " leaf tasks"
                << std::endl;
    }


//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
  template <typename EntGenerator>
  void compress(const EntGenerator& e_gen) {
    const auto nclusters = rta_.nodes().size();

    // 1: INITIALIZE QUEUE WITH LEAF TASKS
    // Leaf tasks have no dependencies, so they're ready to compute immediately
    for (auto* leaf : leaf_tasks_) {
      task_queue_.enqueue(leaf);
    }
    // At this point:
    //   - The queue contains all leaf tasks
    //   - All threads are about to start working

    // 2: PARALLEL PROCESSING
    // Each thread repeatedly:
    // 1. Dequeue a ready task
    // 2. Compute its block
    // 3. Notify parents (which may become ready)
    // 4. Garbage collect the task's children if possible

#pragma omp parallel
    {
      while (true) {
        // Get next task (blocks if queue empty but work remains)
        MatrixBlockTask* task = task_queue_.dequeue();

        // nullptr means all tasks are done
        if (!task) break;

        // --- COMPUTE THE BLOCK ---
        computeBlock(task, e_gen, nclusters);

        // --- NOTIFY PARENTS ---
        // tell the parents that I'm done and you can use my data now
        // If all of a parent's children are done, parent gets enqueued
        notifyParents(task);

        // --- GARBAGE COLLECT CHILDREN ---
        garbageCollectChildren(task);

        // --- MARK AS COMPLETED ---
        task_queue_.mark_completed();
      }
    }
  }


//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
  /**
   * \brief Creates a posteriori thresholded triplets and stores them in the
   * triplet list
   */
  const std::vector<Triplet<Scalar>>& triplets() {
    triplet_list_.clear();

    // Convert unordered_map to vector for OpenMP iteration
    std::vector<MatrixBlockTask*> all_tasks_vec;
    all_tasks_vec.reserve(task_graph_.size());
    for (auto& [key, task] : task_graph_) {
      all_tasks_vec.push_back(task.get());
    }

#pragma omp parallel
    {
      std::vector<Triplet<Scalar>> local_list;

      // Now we can use OpenMP for on a vector
#pragma omp for schedule(dynamic) nowait
      for (size_t i = 0; i < all_tasks_vec.size(); ++i) {
        const auto* task = all_tasks_vec[i];
        const Derived* pr = task->row_cluster;
        const Derived* pc = task->col_cluster;

        // Determine which part of the block to store based on whether
        // clusters are root or not
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

    // Free memory: clear all blocks
    for (auto& [key, task] : task_graph_) {
      task->block.resize(0, 0);
    }

    return triplet_list_;
  }

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
  std::vector<Eigen::Triplet<Scalar>> a_priori_pattern_triplets() {
    std::vector<Eigen::Triplet<Scalar>> retval;

    // Convert unordered_map to vector for OpenMP iteration
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

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
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

    // Make sure the matrix contains the diagonal
    for (Index i = cut_off; i < 17; ++i)
      for (const auto& it : buckets[i])
        if (triplet_list_[it].row() == triplet_list_[it].col())
          retval.push_back(triplet_list_[it]);

    retval.shrink_to_fit();
    return retval;
  }

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
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
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------

  std::vector<Eigen::Triplet<Scalar>> release_triplets() {
    std::vector<Eigen::Triplet<Scalar>> retval;
    std::swap(triplet_list_, retval);
    return retval;
  }

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
 private:
  /**
   * \brief Compute the matrix block for a given task
   *
   * This follows the original 4-case logic:
   * - Case 3 (leaf, leaf): compute directly
   * - Case 1 (noleaf, leaf): recycle from row children
   * - Case 2,0 (*, noleaf): recycle from column children
   */
  template <typename EntGenerator>
  void computeBlock(MatrixBlockTask* task, const EntGenerator& e_gen,
                    size_t nclusters) {
    const Derived* pr = task->row_cluster;
    const Derived* pc = task->col_cluster;

    // Determine which case we're in based on whether clusters have sons
    const char the_case = 2 * (!pr->nSons()) + (!pc->nSons());

    switch (the_case) {
      case 3: {
        // ===================================================================
        // CASE 3: (leaf, leaf) - Both clusters are leaves
        // ===================================================================
        // No children to recycle from, compute directly
        task->block = recursivelyComputeBlock(*pr, *pc, e_gen);
        break;
      }

      case 1: {
        // ===================================================================
        // CASE 1: (noleaf, leaf) - Row has children, column is leaf
        // ===================================================================
        // We recycle blocks from row children (same column, child rows)

        task->block.resize(pr->Q().rows(), pc->Q().cols());
        Index offset = 0;

        for (int k = 0; k < pr->nSons(); ++k) {
          const Index nscalfs = pr->sons(k).nscalfs();
          const size_t child_key =
              pr->sons(k).block_id() + nclusters * pc->block_id();

          if (auto it = task_graph_.find(child_key); it != task_graph_.end()) {
            // Child exists in graph, recycle its block
            const Matrix& child_block = it->second->block;
            task->block.middleRows(offset, nscalfs) =
                child_block.topRows(nscalfs);
          } else {
            // Child doesn't exist (admissible), compute directly
            const Matrix ret = recursivelyComputeBlock(pr->sons(k), *pc, e_gen);
            task->block.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
          }
          offset += nscalfs;
        }

        // Apply the row transformation matrix Q
        task->block = pr->Q().transpose() * task->block;
        break;
      }

      case 2:    // (leaf, noleaf)
      case 0: {  // (noleaf, noleaf)
        // ===================================================================
        // CASE 0,2: Column has children (row may or may not)
        // ===================================================================
        // We recycle blocks from column children (same row, child columns)

        task->block.resize(pr->Q().cols(), pc->Q().rows());
        Index offset = 0;

        for (int k = 0; k < pc->nSons(); ++k) {
          const Index nscalfs = pc->sons(k).nscalfs();
          const size_t child_key =
              pc->sons(k).block_id() * nclusters + pr->block_id();

          if (auto it = task_graph_.find(child_key); it != task_graph_.end()) {
            // Child exists in graph, recycle its block
            const Matrix& child_block = it->second->block;
            task->block.middleCols(offset, nscalfs) =
                child_block.leftCols(nscalfs);
          } else {
            // Child doesn't exist (admissible), compute directly
            const Matrix ret = recursivelyComputeBlock(*pr, pc->sons(k), e_gen);
            task->block.middleCols(offset, nscalfs) = ret.leftCols(nscalfs);
          }
          offset += nscalfs;
        }

        // Apply the column transformation matrix Q
        task->block = task->block * pc->Q();
        break;
      }
    }

    // Mark as computed
    task->is_computed.store(true);
  }

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
  /**
   * \brief Notify parent tasks that this child has completed
   *
   * When a task finishes:
   * 1. Increment children_completed counter of each parent
   * 2. If parent's all children are done, enqueue it as ready
   * 3. Increment parents_notified counter of this task for GC
   */
  void notifyParents(MatrixBlockTask* task) {
    // Notify first parent
    if (task->parent1) {
      updateParentDependency(task->parent1, task);
    }

    // Notify second parent (if exists and different from first)
    if (task->parent2 && task->parent2 != task->parent1) {
      updateParentDependency(task->parent2, task);
    }
  }

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
  /**
   * \brief Update a parent's dependency counter and enqueue if ready
   */
  void updateParentDependency(MatrixBlockTask* parent, MatrixBlockTask* child) {
    // Atomically increment the children_completed counter
    int completed = parent->children_completed.fetch_add(1) + 1;

    // If all children are done, this parent is ready to compute
    if (completed == parent->total_children) {
      task_queue_.enqueue(parent);
    }

    // Increment the child's parents_notified counter
    // (child needs to know when all parents are done using its block)
    child->parents_notified.fetch_add(1);
  }

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
  /**
   * \brief Garbage collect children blocks to save memory
   *
   * After computing a task, we check if we can reduce memory of children:
   * - A child's block can be reduced when ALL its parents have been notified
   * - We keep only the samplet portion (bottom-right corner)
   */
  void garbageCollectChildren(MatrixBlockTask* task) {
    // for (auto* child : task->children) {
    //   // Check if all parents have been notified
    //   int notified = child->parents_notified.load();

    //   if (notified == child->total_parents && child->is_computed.load() &&
    //       !child->memory_reduced.exchange(true)) {
    //     // Reduce memory: keep only samplets if not root
    //     if (!child->row_cluster->is_root() && !child->col_cluster->is_root())
    //     {
    //       child->block = child->block
    //                          .bottomRightCorner(child->row_cluster->nsamplets(),
    //                                             child->col_cluster->nsamplets())
    //                          .eval();
    //     }
    //   }
    // }
  }


//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
  /**
   * \brief Recursively compute a matrix block from the kernel
   *
   * This is the original recursive computation that either:
   * - Uses low-rank approximation if clusters are admissible
   * - Computes dense block if both are leaves
   * - Recurses on children otherwise
   */
  template <typename EntryGenerator>
  Matrix recursivelyComputeBlock(const Derived& TR, const Derived& TC,
                                 const EntryGenerator& e_gen) {
    Matrix buf(0, 0);

    // Check for admissibility
    if (ClusterComparison::compare(TR, TC, eta_) == LowRank) {
      e_gen.interpolate_kernel(TR, TC, &buf);
      return TR.V().transpose() * buf * TC.V();
    } else {
      const char the_case = 2 * (!TR.nSons()) + !TC.nSons();

      switch (the_case) {
        case 3:
          // Both are leafs: compute the block and return
          e_gen.compute_dense_block(TR, TC, &buf);
          return TR.Q().transpose() * buf * TC.Q();

        case 2:
          // Row is leaf, recurse on column
          for (auto j = 0; j < TC.nSons(); ++j) {
            const Index nscalfs = TC.sons(j).nscalfs();
            Matrix ret = recursivelyComputeBlock(TR, TC.sons(j), e_gen);
            buf.conservativeResize(ret.rows(), buf.cols() + nscalfs);
            buf.rightCols(nscalfs) = ret.leftCols(nscalfs);
          }
          return buf * TC.Q();

        case 1:
          // Column is leaf, recurse on row
          for (auto i = 0; i < TR.nSons(); ++i) {
            const Index nscalfs = TR.sons(i).nscalfs();
            Matrix ret = recursivelyComputeBlock(TR.sons(i), TC, e_gen);
            buf.conservativeResize(ret.cols(), buf.cols() + nscalfs);
            buf.rightCols(nscalfs) = ret.transpose().leftCols(nscalfs);
          }
          return (buf * TR.Q()).transpose();

        case 0:
          // Neither is leaf, recurse on both
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

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
  /**
   * \brief Store a matrix block as triplets with optional thresholding
   */
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

  /**
   * \brief Store empty block pattern (for a priori sparsity pattern)
   */
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

  // Task graph: all blocks indexed by their key
  std::unordered_map<size_t, std::unique_ptr<MatrixBlockTask>> task_graph_;

  // Leaf tasks: ready to compute immediately
  std::vector<MatrixBlockTask*> leaf_tasks_;

  // Thread-safe queue for work distribution
  TaskQueue task_queue_;
};
}  // namespace internal
}  // namespace FMCA

#endif