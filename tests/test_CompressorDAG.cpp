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
#include <unordered_map>
#include <vector>

#include "../FMCA/Samplets"
#include "../FMCA/src/util/CompressorDAG.h"
#include "../FMCA/src/util/Graph.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 1000
#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using BallCompare = FMCA::CompareCluster;   // not monotone under nesting
using BoxCompare = FMCA::CompareClusterBB;  // monotone under nesting
template <typename Comparison>
using DAG = FMCA::internal::CompressorDAG<H2SampletTree, Comparison>;
using PatternIdx = std::ptrdiff_t;

struct Checker {
  std::size_t fails = 0;
  void operator()(bool ok, const char *what, PatternIdx k) {
    if (!ok && fails++ < 20)
      std::cout << "FAIL: " << what << " at node " << k << "\n";
  }
};

// checks pattern, dads, sons, flags and counters of a wired DAG. monotone
// states that Comparison is monotone under nesting, in which case the walk
// pattern is the full non-low-rank set and every row dad exists
template <typename Comparison>
bool check_dag(const DAG<Comparison> &dag, const H2SampletTree &TR,
               const H2SampletTree &TC, FMCA::Scalar eta, bool sym,
               bool monotone) {
  using Node = typename DAG<Comparison>::Node;
  using Strategy = typename DAG<Comparison>::Strategy;
  const std::vector<Node> &nodes = dag.nodes();
  const std::size_t m = dag.brows();
  Checker chk;
  auto key = [m](FMCA::Index r, FMCA::Index c) {
    return std::size_t(r) + m * std::size_t(c);
  };
  // (r,c) -> node index, also detects duplicates
  std::unordered_map<std::size_t, PatternIdx> lookup;
  lookup.reserve(nodes.size());
  for (PatternIdx k = 0; k < PatternIdx(nodes.size()); ++k) {
    const Node &v = nodes[k];
    chk(v.pr != nullptr && v.pc != nullptr, "null pr/pc", k);
    chk(lookup.emplace(key(v.pr->block_id(), v.pc->block_id()), k).second,
        "duplicate node", k);
  }
  auto find = [&](FMCA::Index r, FMCA::Index c) {
    const auto it = lookup.find(key(r, c));
    return it == lookup.end() ? PatternIdx(-1) : it->second;
  };
  // completeness against the walk's definition: for fixed pr, (pr, pc) is
  // present iff compare(pr, pc) != LowRank, pc is the root or (pr, pc.dad)
  // is present before filtering, and the filter passes. No monotonicity of
  // the comparison is assumed. reach[c] is the unfiltered presence per pr,
  // level order guarantees dads before sons
  {
    FMCA::internal::RandomTreeAccessor<H2SampletTree> r_rta(TR,
                                                            TR.block_size());
    FMCA::internal::RandomTreeAccessor<H2SampletTree> c_rta(TC,
                                                            TC.block_size());
    std::vector<char> reach(c_rta.nodes().size(), 0);
    std::size_t expected = 0;
    for (const H2SampletTree *pr : r_rta.nodes()) {
      for (const H2SampletTree *pc : c_rta.nodes()) {
        const FMCA::Index c = pc->block_id();
        reach[c] = pr->Q().size() && pc->Q().size() &&
                   (pc->is_root() || reach[pc->dad().block_id()]) &&
                   Comparison::compare(*pr, *pc, eta) != FMCA::LowRank;
        const bool present = reach[c] && (!sym || c >= pr->block_id());
        expected += present;
        chk((find(pr->block_id(), c) >= 0) == present, "pattern entry", -1);
      }
    }
    chk(expected == nodes.size(), "completeness", -1);
  }
  std::size_t n_row_orphans = 0;
  for (PatternIdx k = 0; k < PatternIdx(nodes.size()); ++k) {
    const Node &v = nodes[k];
    const H2SampletTree *pr = v.pr;
    const H2SampletTree *pc = v.pc;
    // pattern
    chk(Comparison::compare(*pr, *pc, eta) != FMCA::LowRank, "admissible node",
        k);
    if (sym) chk(pc->block_id() >= pr->block_id(), "filter", k);
    // dads: topological order, existence, identity. The col dad is present
    // by construction of the walk (up to the filter); the row dad only if
    // (pr.dad, pc) happens to be in the pattern
    chk(v.row_dad < k && v.col_dad < k, "dad not before son", k);
    if (pr->is_root()) {
      chk(v.row_dad < 0, "root has row dad", k);
    } else {
      const PatternIdx e = find(pr->dad().block_id(), pc->block_id());
      chk(v.row_dad == e, "row dad link", k);
      n_row_orphans += (e < 0);
    }
    if (pc->is_root()) {
      chk(v.col_dad < 0, "root has col dad", k);
    } else {
      const bool must = !sym || pc->dad().block_id() >= pr->block_id();
      chk((v.col_dad >= 0) == must, "col dad presence", k);
      if (v.col_dad >= 0)
        chk(nodes[v.col_dad].pc == std::addressof(pc->dad()) &&
                nodes[v.col_dad].pr == pr,
            "col dad wrong", k);
    }
    // strategy and son slots, with back-links
    const FMCA::Index nrs = pr->nSons();
    const FMCA::Index ncs = pc->nSons();
    switch (v.strategy) {
      case Strategy::Leaf:
        chk(!nrs && !ncs, "Leaf on inner pair", k);
        chk(v.sons.empty(), "Leaf has sons", k);
        break;
      case Strategy::Rows:
        chk(nrs > 0, "Rows on row leaf", k);
        chk(v.sons.size() == nrs, "Rows son count", k);
        // forced when pc is a leaf; filtered sons are then recomputed
        if (sym && nrs && ncs)
          chk(pr->sons(nrs - 1).block_id() <= pc->block_id(),
              "Rows with filtered son", k);
        for (FMCA::Index i = 0; i < nrs && i < v.sons.size(); ++i) {
          const PatternIdx e = find(pr->sons(i).block_id(), pc->block_id());
          chk(v.sons[i] == e, "row son slot", k);
          if (e >= 0)
            chk(nodes[e].row_dad == k && nodes[e].row_consumer,
                "row son back-link", k);
        }
        break;
      case Strategy::Cols:
        chk(ncs > 0, "Cols on col leaf", k);
        chk(v.sons.size() == ncs, "Cols son count", k);
        for (FMCA::Index i = 0; i < ncs && i < v.sons.size(); ++i) {
          const PatternIdx e = find(pr->block_id(), pc->sons(i).block_id());
          chk(v.sons[i] == e, "col son slot", k);
          if (e >= 0)
            chk(nodes[e].col_dad == k && nodes[e].col_consumer,
                "col son back-link", k);
        }
        break;
    }
    // flags are exactly "my dad on that side chose that side"
    chk(v.row_consumer ==
            (v.row_dad >= 0 && nodes[v.row_dad].strategy == Strategy::Rows),
        "row_consumer flag", k);
    chk(v.col_consumer ==
            (v.col_dad >= 0 && nodes[v.col_dad].strategy == Strategy::Cols),
        "col_consumer flag", k);
    // counters
    FMCA::Index present = 0;
    for (PatternIdx s : v.sons) present += (s >= 0);
    chk(v.deps.load() == present, "deps counter", k);
    chk(v.consumers.load() == FMCA::Index(v.row_consumer + v.col_consumer + 1),
        "consumers counter", k);
  }
  if (monotone)
    chk(n_row_orphans == 0, "row orphan under monotone comparison", -1);
  std::cout << (sym ? "sym  " : "unsym") << " nodes = " << nodes.size()
            << ", row orphans = " << n_row_orphans
            << (chk.fails ? "  FAILED" : "  passed") << std::endl;
  return chk.fails == 0;
}

// DAG plot on the (pc.id, pr.id) grid, edges to both dads, weight 2 on the
// edges the schedule uses (consumer side), 1 otherwise
template <typename Comparison>
void plot_dag(const DAG<Comparison> &dag, const std::string &fname) {
  using Node = typename DAG<Comparison>::Node;
  const std::vector<Node> &nodes = dag.nodes();
  std::vector<Eigen::Triplet<double>> tvec;
  tvec.reserve(4 * nodes.size());
  FMCA::Matrix PG(2, nodes.size());
  for (PatternIdx k = 0; k < PatternIdx(nodes.size()); ++k) {
    const Node &v = nodes[k];
    PG(0, k) = v.pc->block_id();
    PG(1, k) = v.pr->block_id();
    if (v.row_dad >= 0) {
      const double w = v.row_consumer ? 2. : 1.;
      tvec.emplace_back(k, v.row_dad, w);
      tvec.emplace_back(v.row_dad, k, w);
    }
    if (v.col_dad >= 0) {
      const double w = v.col_consumer ? 2. : 1.;
      tvec.emplace_back(k, v.col_dad, w);
      tvec.emplace_back(v.col_dad, k, w);
    }
  }
  FMCA::Graph<int, double> G;
  G.init(nodes.size(), tvec);
  G.print(fname, PG, false);
}

template <typename Comparison>
bool run(const H2SampletTree &hst1, const H2SampletTree &hst2, FMCA::Scalar eta,
         bool monotone, const std::string &tag) {
  FMCA::Tictoc T;
  bool ok = true;
  DAG<Comparison> dag;
  std::cout << "--- " << tag << std::endl;
  T.tic();
  dag.init(hst1, hst1, eta, true);
  T.toc("dag init sym:   ");
  std::cout << "block matrix size: " << dag.brows() << "x" << dag.bcols()
            << std::endl;
  ok &= check_dag(dag, hst1, hst1, eta, true, monotone);
  // plot_dag(dag, "dag_sym_" + tag + ".vtk");

  T.tic();
  dag.init(hst1, hst1, eta, false);
  T.toc("dag init unsym: ");
  std::cout << "block matrix size: " << dag.brows() << "x" << dag.bcols()
            << std::endl;
  ok &= check_dag(dag, hst1, hst1, eta, false, monotone);

  T.tic();
  dag.init(hst1, hst2, eta, false);
  T.toc("dag init rect:  ");
  std::cout << "block matrix size: " << dag.brows() << "x" << dag.bcols()
            << std::endl;
  ok &= check_dag(dag, hst1, hst2, eta, false, monotone);
  // plot_dag(dag, "dag_rect_" + tag + ".vtk");
  return ok;
}

int main() {
  const FMCA::Matrix P1 = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  const FMCA::Matrix P2 =
      0.5 * (FMCA::Matrix::Random(DIM, NPTS + 4000).array() + 1);
  const FMCA::Scalar eta = 0.99;
  const FMCA::Index dtilde = 4;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const Moments mom1(P1, mpole_deg);
  const SampletMoments samp_mom1(P1, dtilde - 1);
  const Moments mom2(P2, mpole_deg);
  const SampletMoments samp_mom2(P2, dtilde - 1);
  H2SampletTree hst1(mom1, samp_mom1, 0, P1);
  H2SampletTree hst2(mom2, samp_mom2, 0, P2);

  bool ok = true;
  ok &= run<BallCompare>(hst1, hst2, eta, false, "ball");
  ok &= run<BoxCompare>(hst1, hst2, eta, true, "box");
  return ok ? 0 : 1;
}
