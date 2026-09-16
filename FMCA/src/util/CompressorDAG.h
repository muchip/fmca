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
#include "RandomTreeAccessor.h"

namespace FMCA {
namespace internal {

template <typename H2STreeType, typename ClusterComparison = CompareCluster>
class CompressorDAG {
 public:
  using PatternIdx = std::ptrdiff_t;
  using Pattern = Eigen::SparseMatrix<PatternIdx, Eigen::ColMajor, PatternIdx>;
  using PatternTriplet = Eigen::Triplet<PatternIdx, PatternIdx>;
  enum class Strategy { Rows, Cols, Leaf };
  struct Node {
    const H2STreeType *pr = nullptr;
    const H2STreeType *pc = nullptr;
    PatternIdx row_dad = -1;          // (pr.dad, pc), -1 if absent
    PatternIdx col_dad = -1;          // (pr, pc.dad), -1 if absent
    std::vector<PatternIdx> sons;     // chosen side only, -1 = absent
    Matrix block;                     // freed by Matrix().swap(block)
    std::atomic<Index> deps{0};       // present sons on chosen side
    std::atomic<Index> consumers{0};  // row_c + col_c + 1
    Strategy strategy = Strategy::Leaf;
    bool row_consumer = false;  // row_dad chose Rows
    bool col_consumer = false;  // col_dad chose Cols
  };
  CompressorDAG() : block_rows_(0), block_cols_(0), sym_(false) {}
  void init(const H2STreeType &TR, const H2STreeType &TC, Scalar eta,
            bool sym = false) {
    sym &= (std::addressof(TR.derived()) == std::addressof(TC.derived()));
    sym_ = sym;
    r_rta_.init(TR, TR.block_size());
    if (!sym) c_rta_.init(TC, TC.block_size());
    const RandomTreeAccessor<H2STreeType> &r_rta = r_rta_;
    const RandomTreeAccessor<H2STreeType> &c_rta = sym ? r_rta : c_rta_;
    const PatternIdx m = r_rta.nodes().size();
    const PatternIdx n = c_rta.nodes().size();
    // col_dads[k] is the walk index of (pr, pc.dad) for the node with walk
    // index k, -1 if absent. The pattern value is the walk index itself
    std::vector<PatternIdx> col_dads;
    {
      std::vector<PatternTriplet> triplets;
      triplets.reserve(std::max(m, n) *
                       std::ceil(std::log(std::min(m, n) + 2)));
      col_dads.reserve(triplets.capacity());
      std::vector<std::pair<const H2STreeType *, PatternIdx>> col_stack;
      PatternIdx k = 0;
      for (const H2STreeType *pr : r_rta.nodes()) {
        col_stack.assign(
            1, std::make_pair(std::addressof(TC.derived()), PatternIdx(-1)));
        while (!col_stack.empty()) {
          const H2STreeType *pc = col_stack.back().first;
          const PatternIdx col_dad = col_stack.back().second;
          col_stack.pop_back();
          PatternIdx self = -1;
          if (!sym || pc->block_id() >= pr->block_id()) {
            self = k++;
            triplets.emplace_back(pr->block_id(), pc->block_id(), self);
            col_dads.push_back(col_dad);
          }
          for (Index i = 0; i < pc->nSons(); ++i)
            if (ClusterComparison::compare(*pr, pc->sons(i), eta) != LowRank)
              col_stack.emplace_back(std::addressof(pc->sons(i)), self);
        }
      }
      Pattern().swap(pattern_);
      pattern_.resize(m, n);
      // the DFS visits each (pr, pc) at most once, so no duplicates are ever
      // merged and pattern.nonZeros() == k. The functor only documents this
      pattern_.setFromTriplets(
          triplets.begin(), triplets.end(),
          [](const PatternIdx &, const PatternIdx &b) { return b; });
    }
    std::vector<Node>(pattern_.nonZeros()).swap(node_storage_);
    wire(pattern_, col_dads, r_rta, c_rta, sym);
    block_rows_ = m;
    block_cols_ = n;
    return;
  }
  const std::vector<Node> &nodes() const { return node_storage_; }

  std::vector<Node> &nodes() { return node_storage_; }
  const Pattern &pattern() const { return pattern_; }
  const RandomTreeAccessor<H2STreeType> &r_rta() const { return r_rta_; }
  const RandomTreeAccessor<H2STreeType> &c_rta() const {
    return sym_ ? r_rta_ : c_rta_;
  }
  bool sym() const { return sym_; }
  Index brows() const { return block_rows_; }
  Index bcols() const { return block_cols_; }

 private:
  void wire(const Pattern &pattern, const std::vector<PatternIdx> &col_dads,
            const RandomTreeAccessor<H2STreeType> &r_rta,
            const RandomTreeAccessor<H2STreeType> &c_rta, bool sym) {
    const PatternIdx *outer = pattern.outerIndexPtr();
    const PatternIdx *inner = pattern.innerIndexPtr();
    const PatternIdx *val = pattern.valuePtr();
    const std::vector<PatternIdx> &rdad = r_rta.dad();
    const std::vector<PatternIdx> &rpos = r_rta.child_pos();
    const std::vector<PatternIdx> &cpos = c_rta.child_pos();
    // pass 1: CCS sweep. Sets pr, pc and both dads.
    // present son counts stored in the atomics
    // (row count in deps, col count in consumers).
    std::vector<PatternIdx> w(pattern.rows(), -1);
    for (PatternIdx c = 0; c < pattern.cols(); ++c) {
      const H2STreeType *pc = c_rta.nodes()[c];
      for (PatternIdx p = outer[c]; p < outer[c + 1]; ++p) {
        const PatternIdx r = inner[p];
        const PatternIdx self = val[p];
        Node &node = node_storage_[self];
        node.pr = r_rta.nodes()[r];
        node.pc = pc;
        node.col_dad = col_dads[self];
        if (node.col_dad >= 0)
          node_storage_[node.col_dad].consumers.fetch_add(
              1, std::memory_order_relaxed);
        if (r) {
          node.row_dad = w[rdad[r]];
          if (node.row_dad >= 0)
            node_storage_[node.row_dad].deps.fetch_add(
                1, std::memory_order_relaxed);
        }
        w[r] = self;
      }
      for (PatternIdx p = outer[c]; p < outer[c + 1]; ++p) w[inner[p]] = -1;
    }
    // pass 2: assign strategy and set consumer flags. In walk order both
    // dads precede their sons, so the dad's side is known when the son
    // registers with it
    for (PatternIdx k = 0; k < PatternIdx(node_storage_.size()); ++k) {
      Node &node = node_storage_[k];
      const Index nrc = node.pr->nSons();
      const Index ncc = node.pc->nSons();
      const Index rcp = node.deps.load(std::memory_order_relaxed);
      const Index ccp = node.consumers.load(std::memory_order_relaxed);
      // fix strategy
      // a row son (pr.son, pc) is filtered iff sym and pc.id < pr.son.id;
      // son ids are contiguous, so the last son decides
      const bool row_filtered =
          sym && nrc && node.pc->block_id() < node.pr->sons(nrc - 1).block_id();
      Index present = 0;
      if (!nrc && !ncc) {
        node.strategy = Strategy::Leaf;
      } else if (!ncc || (nrc && !row_filtered && nrc - rcp < ncc - ccp)) {
        node.strategy = Strategy::Rows;
        node.sons.assign(nrc, -1);
        present = rcp;
      } else {
        node.strategy = Strategy::Cols;
        node.sons.assign(ncc, -1);
        present = ccp;
      }
      node.deps.store(present, std::memory_order_relaxed);
      if (node.row_dad >= 0 &&
          node_storage_[node.row_dad].strategy == Strategy::Rows) {
        node_storage_[node.row_dad].sons[rpos[node.pr->block_id()]] = k;
        node.row_consumer = true;
      }
      if (node.col_dad >= 0 &&
          node_storage_[node.col_dad].strategy == Strategy::Cols) {
        node_storage_[node.col_dad].sons[cpos[node.pc->block_id()]] = k;
        node.col_consumer = true;
      }
      node.consumers.store(node.row_consumer + node.col_consumer + 1,
                           std::memory_order_relaxed);
    }
    return;
  }
  Pattern pattern_;
  RandomTreeAccessor<H2STreeType> r_rta_;
  RandomTreeAccessor<H2STreeType> c_rta_;
  std::vector<Node> node_storage_;
  Index block_rows_;
  Index block_cols_;
  bool sym_;
};

}  // namespace internal
}  // namespace FMCA
#endif
