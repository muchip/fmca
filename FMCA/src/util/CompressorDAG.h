// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_UTIL_COMPRESSORDAG_H_
#define FMCA_UTIL_COMPRESSORDAG_H_

#include "Macros.h"

namespace FMCA {
namespace internal {

template <typename H2STreeType, typename PayloadType,
          typename ClusterComparison = CompareCluster>
class CompressorDAG {
 public:
  struct Node {
    enum RecyclingStrategy { Rows, Cols, Leaf };
    const H2STreeType *pr = nullptr;
    const H2STreeType *pc = nullptr;
    std::vector<Node *> row_sons;
    std::vector<Node *> col_sons;
    Node *row_dad = nullptr;
    Node *col_dad = nullptr;
    // scheduling related fields
    RecyclingStrategy strategy = Leaf;
    PayloadType block{nullptr, 0, 0};
    std::atomic<Index> deps_remaining{0};
    std::atomic<Index> consumers_remaining{0};
  };

  using Pattern = Eigen::SparseMatrix<Node *, Eigen::ColMajor, std::ptrdiff_t>;
  using PatternTriplet = Eigen::Triplet<Node *>;

  void init(const H2STreeType &TR, const H2STreeType &TC, Scalar eta,
            bool sym = false) {
    sym &= (std::addressof(TR.derived()) == std::addressof(TC.derived()));
    RandomTreeAccessor<H2STreeType> r_rta(TR, TR.block_size());
    Pattern pattern;
    std::vector<PatternTriplet> triplets;
    node_storage_.clear();
    if (sym) {
      const std::ptrdiff_t n = r_rta.nodes().size();
      triplets.reserve(n * std::ceil(std::log(n + 2)));
      for (Index j = 0; j < r_rta.nodes().size(); ++j) {
        const H2STreeType *pr = r_rta.nodes()[j];
        std::vector<std::pair<const H2STreeType *, Node *>> col_stack;
        col_stack.push_back(
            std::make_pair(std::addressof(TC.derived()), nullptr));
        while (!col_stack.empty()) {
          const H2STreeType *pc = col_stack.back().first;
          Node *col_dad_ptr = col_stack.back().second;
          col_stack.pop_back();
          Node *this_ptr = nullptr;
          if (pc->block_id() >= pr->block_id()) {
            node_storage_.emplace_back();
            this_ptr = std::addressof(node_storage_.back());
            this_ptr->pr = pr;
            this_ptr->pc = pc;
            this_ptr->col_sons.assign(pc->nSons(), nullptr);
            this_ptr->row_sons.assign(pr->nSons(), nullptr);
            this_ptr->col_dad = col_dad_ptr;
            if (col_dad_ptr != nullptr)
              col_dad_ptr->col_sons[r_rta.child_pos()[pc->block_id()]] =
                  this_ptr;
            triplets.emplace_back(pr->block_id(), pc->block_id(), this_ptr);
          }
          for (Index i = 0; i < pc->nSons(); ++i)
            if (ClusterComparison::compare(*pr, pc->sons(i), eta) != LowRank)
              col_stack.push_back(
                  std::make_pair(std::addressof(pc->sons(i)), this_ptr));
        }
      }
      pattern = Pattern(n, n);
      block_rows_ = n;
      block_cols_ = n;
    } else {
      RandomTreeAccessor<H2STreeType> c_rta(TC, TC.block_size());
      const std::ptrdiff_t m = r_rta.nodes().size();
      const std::ptrdiff_t n = c_rta.nodes().size();
      triplets.reserve(std::max(n, m) *
                       std::ceil(std::log(std::min(m, n) + 2)));
      for (Index j = 0; j < r_rta.nodes().size(); ++j) {
        const H2STreeType *pr = r_rta.nodes()[j];
        std::vector<std::pair<const H2STreeType *, Node *>> col_stack;
        col_stack.push_back(
            std::make_pair(std::addressof(TC.derived()), nullptr));
        while (!col_stack.empty()) {
          const H2STreeType *pc = col_stack.back().first;
          Node *col_dad_ptr = col_stack.back().second;
          col_stack.pop_back();
          Node *this_ptr = nullptr;
          {
            node_storage_.emplace_back();
            this_ptr = std::addressof(node_storage_.back());
            this_ptr->pr = pr;
            this_ptr->pc = pc;
            this_ptr->col_sons.assign(pc->nSons(), nullptr);
            this_ptr->row_sons.assign(pr->nSons(), nullptr);
            this_ptr->col_dad = col_dad_ptr;
            if (col_dad_ptr != nullptr)
              col_dad_ptr->col_sons[c_rta.child_pos()[pc->block_id()]] =
                  this_ptr;
            triplets.emplace_back(pr->block_id(), pc->block_id(), this_ptr);
          }
          for (Index i = 0; i < pc->nSons(); ++i)
            if (ClusterComparison::compare(*pr, pc->sons(i), eta) != LowRank)
              col_stack.push_back(
                  std::make_pair(std::addressof(pc->sons(i)), this_ptr));
        }
      }
      pattern = Pattern(m, n);
      block_rows_ = m;
      block_cols_ = n;
    }
    pattern.setFromTriplets(triplets.begin(), triplets.end(),
                            [](Node *a, Node *b) { return b; });
    wire_rows(pattern, r_rta);

    return;
  }

  const std::deque<Node> &nodes() const { return node_storage_; }

  std::deque<Node> &nodes() { return node_storage_; }

  Index brows() const { return block_rows_; }
  Index bcols() const { return block_cols_; }

 private:
  static void wire_rows(Pattern &pattern,
                        const RandomTreeAccessor<H2STreeType> &r_rta) {
    const std::ptrdiff_t *outer = pattern.outerIndexPtr();
    const std::ptrdiff_t *inner = pattern.innerIndexPtr();
    Node *const *val = pattern.valuePtr();
    const std::ptrdiff_t n = pattern.cols();
    std::vector<Node *> w(pattern.rows(), nullptr);

    for (std::ptrdiff_t col = 0; col < n; ++col) {
      for (std::ptrdiff_t k = outer[col]; k < outer[col + 1]; ++k) {
        Node *node = val[k];
        const H2STreeType *pr = node->pr;

        if (!pr->is_root()) {
          Node *dad_node = w[pr->dad().block_id()];
          if (dad_node != nullptr) {
            node->row_dad = dad_node;
            dad_node->row_sons[r_rta.child_pos()[pr->block_id()]] = node;
          }
        }
        w[pr->block_id()] = node;
      }
      // reset only the touched slots to avoid an O(rows) clear per column
      for (std::ptrdiff_t k = outer[col]; k < outer[col + 1]; ++k)
        w[inner[k]] = nullptr;
    }
    return;
  }

  std::deque<Node> node_storage_;
  Index block_rows_;
  Index block_cols_;
};

}  // namespace internal
}  // namespace FMCA
#endif
