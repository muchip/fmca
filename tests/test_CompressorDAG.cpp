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
#include <set>
#include <sstream>
#include <unordered_set>
#include <utility>
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
using H2SampletTree = FMCA::H2SampletTree<FMCA::UnitKDTree>;
using CompressorDag =
    FMCA::internal::CompressorDAG<H2SampletTree, FMCA::Matrix>;

template <class Dag>
bool diagnose_dag(const Dag &dag, std::ostream &os) {
  using Node = typename Dag::Node;
  bool ok = true;

  auto node_name = [](const Node *n) {
    std::ostringstream ss;
    if (n == nullptr) {
      ss << "(null)";
    } else {
      ss << "(" << n->pr->block_id() << "," << n->pc->block_id() << ")";
    }
    return ss.str();
  };

  auto index_in_vec = [](const std::vector<Node *> &vec, const Node *ptr) {
    FMCA::Index pos = -1;
    FMCA::Index cnt = 0;
    for (FMCA::Index i = 0; i < static_cast<FMCA::Index>(vec.size()); ++i) {
      if (vec[i] == ptr) {
        pos = i;
        ++cnt;
      }
    }
    return std::make_pair(cnt, pos);
  };

  std::unordered_set<const Node *> node_ptrs;
  for (const auto &node : dag.nodes()) node_ptrs.insert(&node);

  std::set<std::pair<FMCA::Index, FMCA::Index>> seen_pairs;

  FMCA::Index n_row_edges = 0;
  FMCA::Index n_col_edges = 0;

  for (const auto &u_ref : dag.nodes()) {
    const Node *u = &u_ref;

    if (u->pr == nullptr || u->pc == nullptr) {
      os << "ERROR: null pr/pc at node " << node_name(u) << "\n";
      ok = false;
      continue;
    }

    const auto key = std::make_pair(u->pr->block_id(), u->pc->block_id());
    if (!seen_pairs.insert(key).second) {
      os << "ERROR: duplicate DAG node " << node_name(u) << "\n";
      ok = false;
    }

    if (u->row_dad == u) {
      os << "ERROR: self row_dad at node " << node_name(u) << "\n";
      ok = false;
    }
    if (u->col_dad == u) {
      os << "ERROR: self col_dad at node " << node_name(u) << "\n";
      ok = false;
    }

    if (static_cast<FMCA::Index>(u->row_sons.size()) != u->pr->nSons()) {
      os << "ERROR: row_sons has wrong size at node " << node_name(u)
         << ": got " << u->row_sons.size() << ", expected " << u->pr->nSons()
         << "\n";
      ok = false;
    }

    if (static_cast<FMCA::Index>(u->col_sons.size()) != u->pc->nSons()) {
      os << "ERROR: col_sons has wrong size at node " << node_name(u)
         << ": got " << u->col_sons.size() << ", expected " << u->pc->nSons()
         << "\n";
      ok = false;
    }

    if (u->row_dad != nullptr) {
      ++n_row_edges;

      if (!node_ptrs.count(u->row_dad)) {
        os << "ERROR: row_dad not in DAG for node " << node_name(u) << " -> "
           << node_name(u->row_dad) << "\n";
        ok = false;
      }

      if (u->row_dad->pc != u->pc) {
        os << "ERROR: row edge does not preserve pc: child " << node_name(u)
           << " dad " << node_name(u->row_dad) << "\n";
        ok = false;
      }

      if (u->pr->is_root()) {
        os << "ERROR: root node has row_dad: " << node_name(u) << "\n";
        ok = false;
      } else if (u->row_dad->pr != std::addressof(u->pr->dad())) {
        os << "ERROR: row_dad has wrong pr: child " << node_name(u) << " dad "
           << node_name(u->row_dad) << "\n";
        ok = false;
      }

      const auto row_loc = index_in_vec(u->row_dad->row_sons, u);
      const FMCA::Index cnt = row_loc.first;
      const FMCA::Index pos = row_loc.second;
      if (cnt != 1) {
        os << "ERROR: row_dad/row_sons mismatch: child " << node_name(u)
           << " found " << cnt << " times in dad " << node_name(u->row_dad)
           << "\n";
        ok = false;
      } else {
        FMCA::Index expected_pos = -1;
        for (FMCA::Index s = 0; s < u->row_dad->pr->nSons(); ++s) {
          if (std::addressof(u->row_dad->pr->sons(s)) == u->pr) {
            expected_pos = s;
            break;
          }
        }
        if (expected_pos < 0) {
          os << "ERROR: could not recover expected row slot for child "
             << node_name(u) << " in dad " << node_name(u->row_dad) << "\n";
          ok = false;
        } else if (pos != expected_pos) {
          os << "ERROR: row child stored in wrong slot: child " << node_name(u)
             << " at slot " << pos << ", expected " << expected_pos << "\n";
          ok = false;
        }
      }
    }

    if (u->col_dad != nullptr) {
      ++n_col_edges;

      if (!node_ptrs.count(u->col_dad)) {
        os << "ERROR: col_dad not in DAG for node " << node_name(u) << " -> "
           << node_name(u->col_dad) << "\n";
        ok = false;
      }

      if (u->col_dad->pr != u->pr) {
        os << "ERROR: col edge does not preserve pr: child " << node_name(u)
           << " dad " << node_name(u->col_dad) << "\n";
        ok = false;
      }

      if (u->pc->is_root()) {
        os << "ERROR: root node has col_dad: " << node_name(u) << "\n";
        ok = false;
      } else if (u->col_dad->pc != std::addressof(u->pc->dad())) {
        os << "ERROR: col_dad has wrong pc: child " << node_name(u) << " dad "
           << node_name(u->col_dad) << "\n";
        ok = false;
      }

      const auto col_loc = index_in_vec(u->col_dad->col_sons, u);
      const FMCA::Index cnt = col_loc.first;
      const FMCA::Index pos = col_loc.second;
      if (cnt != 1) {
        os << "ERROR: col_dad/col_sons mismatch: child " << node_name(u)
           << " found " << cnt << " times in dad " << node_name(u->col_dad)
           << "\n";
        ok = false;
      } else {
        FMCA::Index expected_pos = -1;
        for (FMCA::Index s = 0; s < u->col_dad->pc->nSons(); ++s) {
          if (std::addressof(u->col_dad->pc->sons(s)) == u->pc) {
            expected_pos = s;
            break;
          }
        }
        if (expected_pos < 0) {
          os << "ERROR: could not recover expected col slot for child "
             << node_name(u) << " in dad " << node_name(u->col_dad) << "\n";
          ok = false;
        } else if (pos != expected_pos) {
          os << "ERROR: col child stored in wrong slot: child " << node_name(u)
             << " at slot " << pos << ", expected " << expected_pos << "\n";
          ok = false;
        }
      }
    }

    for (FMCA::Index s = 0; s < static_cast<FMCA::Index>(u->row_sons.size());
         ++s) {
      const Node *v = u->row_sons[s];
      if (v == nullptr) continue;

      if (!node_ptrs.count(v)) {
        os << "ERROR: row_son not in DAG: parent " << node_name(u) << " son "
           << node_name(v) << "\n";
        ok = false;
      }
      if (v->row_dad != u) {
        os << "ERROR: row_son back-link mismatch: parent " << node_name(u)
           << " son " << node_name(v) << "\n";
        ok = false;
      }
      if (v->pc != u->pc) {
        os << "ERROR: row_son edge does not preserve pc: parent "
           << node_name(u) << " son " << node_name(v) << "\n";
        ok = false;
      }
      if (v->pr->is_root()) {
        os << "ERROR: row_son is root unexpectedly: parent " << node_name(u)
           << " son " << node_name(v) << "\n";
        ok = false;
      } else {
        if (std::addressof(v->pr->dad()) != u->pr) {
          os << "ERROR: row_son has wrong row parent: parent " << node_name(u)
             << " son " << node_name(v) << "\n";
          ok = false;
        }
        if (std::addressof(u->pr->sons(s)) != v->pr) {
          os << "ERROR: row_son stored in wrong slot " << s << ": parent "
             << node_name(u) << " son " << node_name(v) << "\n";
          ok = false;
        }
      }
    }

    for (FMCA::Index s = 0; s < static_cast<FMCA::Index>(u->col_sons.size());
         ++s) {
      const Node *v = u->col_sons[s];
      if (v == nullptr) continue;

      if (!node_ptrs.count(v)) {
        os << "ERROR: col_son not in DAG: parent " << node_name(u) << " son "
           << node_name(v) << "\n";
        ok = false;
      }
      if (v->col_dad != u) {
        os << "ERROR: col_son back-link mismatch: parent " << node_name(u)
           << " son " << node_name(v) << "\n";
        ok = false;
      }
      if (v->pr != u->pr) {
        os << "ERROR: col_son edge does not preserve pr: parent "
           << node_name(u) << " son " << node_name(v) << "\n";
        ok = false;
      }
      if (v->pc->is_root()) {
        os << "ERROR: col_son is root unexpectedly: parent " << node_name(u)
           << " son " << node_name(v) << "\n";
        ok = false;
      } else {
        if (std::addressof(v->pc->dad()) != u->pc) {
          os << "ERROR: col_son has wrong col parent: parent " << node_name(u)
             << " son " << node_name(v) << "\n";
          ok = false;
        }
        if (std::addressof(u->pc->sons(s)) != v->pc) {
          os << "ERROR: col_son stored in wrong slot " << s << ": parent "
             << node_name(u) << " son " << node_name(v) << "\n";
          ok = false;
        }
      }
    }
  }

  os << "DAG diagnostics: " << (ok ? "PASSED" : "FAILED") << "\n"
     << "nodes = " << dag.nodes().size()
     << ", unique pairs = " << seen_pairs.size() << "\n"
     << "row edges = " << n_row_edges << ", col edges = " << n_col_edges
     << "\n";

  return ok;
}

int main() {
  FMCA::Tictoc T;
  const FMCA::Matrix P1 = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  const FMCA::Matrix P2 =
      0.5 * (FMCA::Matrix::Random(DIM, NPTS + 100).array() + 1);
  const FMCA::Scalar threshold = 1e-6;
  const FMCA::Scalar eta = 0.99;
  const FMCA::Index dtilde = 4;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const Moments mom1(P1, mpole_deg);
  const SampletMoments samp_mom1(P1, dtilde - 1);
  const Moments mom2(P2, mpole_deg);
  const SampletMoments samp_mom2(P2, dtilde - 1);

  std::cout << "dtilde:                       " << dtilde << std::endl;
  std::cout << "mpole_deg:                    " << mpole_deg << std::endl;
  std::cout << "eta:                          " << eta << std::endl;
  H2SampletTree hst1(mom1, samp_mom1, 0, P1, 3);
  H2SampletTree hst2(mom2, samp_mom2, 0, P2, 3);
  T.tic();
  CompressorDag dag;
  dag.init(hst1, hst1, eta, false);
  T.toc("dag init: ");
  std::cout << dag.brows() << "x" << dag.bcols() << std::endl;
  T.toc();

  const bool ok = diagnose_dag(dag, std::cout);
  if (!ok) {
    std::cerr << "CompressorDAG diagnostics failed.\n";
    return 1;
  }

  std::vector<Eigen::Triplet<double>> tvec;
  for (const auto &it : dag.nodes()) {
    const FMCA::Index node_id =
        (it.pr)->block_id() + dag.brows() * (it.pc)->block_id();

    for (const auto *it2 : it.row_sons) {
      if (it2 == nullptr) continue;
      const FMCA::Index son_id =
          (it2->pr)->block_id() + dag.brows() * (it2->pc)->block_id();
      tvec.emplace_back(node_id, son_id, 1.);
      tvec.emplace_back(son_id, node_id, 1.);
    }

    for (const auto *it2 : it.col_sons) {
      if (it2 == nullptr) continue;
      const FMCA::Index son_id =
          (it2->pr)->block_id() + dag.brows() * (it2->pc)->block_id();
      tvec.emplace_back(node_id, son_id, 1.);
      tvec.emplace_back(son_id, node_id, 1.);
    }
  }

  FMCA::Graph<int, double> G;
  G.init(dag.brows() * dag.bcols(), tvec);
  FMCA::Matrix PG(2, dag.brows() * dag.bcols());
  FMCA::Index k = 0;
  for (FMCA::Index j = 0; j < dag.bcols(); ++j)
    for (FMCA::Index i = 0; i < dag.brows(); ++i) PG.col(k++) << j, i;
  G.print("dag.vtk", PG, false);
  return 0;
}
