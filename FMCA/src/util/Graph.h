// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2024, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_UTIL_GRAPH_H_
#define FMCA_UTIL_GRAPH_H_
//
#include "Macros.h"
#include "graphIO.h"

namespace FMCA {

template <typename IndexT, typename ValueT>
class Graph {
 public:
  using IndexType = IndexT;
  using ValueType = ValueT;
  using GraphType = Eigen::SparseMatrix<ValueType, Eigen::RowMajor, IndexType>;
  Graph() {};
  template <typename T>
  void init(const IndexType nnodes, const T &triplets) {
    A_.resize(nnodes, nnodes);
    labels_.resize(nnodes);
    std::iota(labels_.begin(), labels_.end(), 0);
    A_.setFromTriplets(triplets.begin(), triplets.end());
    A_.makeCompressed();
    return;
  }

  std::vector<IndexType> computeLandmarkNodes(IndexType M) const {
    std::vector<IndexType> landmarks(M > nnodes() ? nnodes() : M);
    std::vector<bool> is_member(nnodes(), false);
    if (landmarks.size() == nnodes()) {
      std::iota(landmarks.begin(), landmarks.end(), 0);
      return landmarks;
    }
    landmarks[0] = rand() % nnodes();
    is_member[landmarks[0]] = true;
    std::vector<ValueType> min_dists = Dijkstra(landmarks[0]);
    for (IndexType found = 1; found < M; ++found) {
      IndexType max_id = landmarks[found - 1];
      ValueType max = 0;
      // look for maximally distant node
      for (IndexType i = 0; i < min_dists.size(); ++i)
        if (not is_member[i] && min_dists[i] > max) {
          max_id = i;
          max = min_dists[i];
        }
      if (is_member[max_id]) {
        for (IndexType i = 0; i < is_member.size(); ++i) {
          if (not is_member[i]) {
            max_id = i;
            break;
          }
        }
      }
      is_member[max_id] = true;
      landmarks[found] = max_id;
      std::vector<ValueType> dists = Dijkstra(max_id);
      for (IndexType i = 0; i < dists.size(); ++i)
        min_dists[i] = min_dists[i] > dists[i] ? dists[i] : min_dists[i];
    }
    return landmarks;
  }

  const IndexType nnodes() const { return A_.rows(); }
  const IndexType nedges() const { return A_.nonZeros(); }
  const std::vector<IndexType> &labels() const { return labels_; }
  std::vector<IndexType> &labels() { return labels_; }
  GraphType &graph() { return A_; }
  const GraphType &graph() const { return A_; }
  Matrix distanceMatrix() const { return FloydWarshall(); }

  template <typename T>
  std::vector<std::vector<ValueType>> partialDistanceMatrix(
      const T &indices) const {
    std::vector<std::vector<ValueType>> retval(indices.size());
#pragma omp parallel for
    for (IndexType i = 0; i < retval.size(); ++i)
      retval[i] = Dijkstra(indices[i]);
    return retval;
  }

  template <typename Derived>
  void print(const std::string &fileName, const MatrixBase<Derived> &P,
             const bool use_labels = true) const {
    Derived Ploc(P.rows(), labels_.size());
    if (use_labels)
      for (IndexType i = 0; i < Ploc.cols(); ++i)
        Ploc.col(i) = P.col(labels_[i]);
    else
      for (IndexType i = 0; i < Ploc.cols(); ++i) Ploc.col(i) = P.col(i);

    IO::plotGraph(fileName, Ploc, A_);
    return;
  }

  template <typename Derived>
  void printSignal(const std::string &fileName, const MatrixBase<Derived> &P,
                   const Vector &sig) const {
    IO::plotGraphSignal(fileName, P, A_, sig);
    return;
  }

 private:
  //////////////////////////////////////////////////////////////////////////////
  Matrix FloydWarshall() const {
    Matrix D = A_;
#pragma omp parallel for
    for (Index j = 0; j < nnodes(); ++j) {
      for (Index i = 0; i < nnodes(); ++i)
        D(i, j) = std::abs(D(i, j)) < FMCA_ZERO_TOLERANCE ? FMCA_INF : D(i, j);
      D(j, j) = 0;
    }
    for (Index k = 0; k < nnodes(); ++k)
#pragma omp parallel for
      for (Index j = 0; j < nnodes(); ++j) {
        const Scalar dkj = D(k, j);
        for (Index i = 0; i < nnodes(); ++i) {
          const Scalar w = D(i, k) + dkj;
          D(i, j) = D(i, j) > w ? w : D(i, j);
        }
      }
    return D;
  }

  std::vector<ValueType> Dijkstra(IndexType node) const {
    using qPair = std::pair<ValueType, IndexType>;
    std::priority_queue<qPair, std::vector<qPair>, std::greater<qPair>> pqueue;
    std::vector<ValueType> dists(nnodes(),
                                 std::numeric_limits<ValueType>::infinity());
    std::vector<bool> settled(nnodes(), false);
    dists[node] = 0;
    pqueue.push(std::make_pair(0., node));
    while (not pqueue.empty()) {
      const ValueType du = pqueue.top().first;
      const IndexType u = pqueue.top().second;
      pqueue.pop();
      if (settled[u]) continue;
      settled[u] = true;
      for (typename GraphType::InnerIterator it(A_, u); it; ++it) {
        const IndexType v = it.col();
        const ValueType w = it.value();
        if (dists[v] > du + w) {
          dists[v] = du + w;
          pqueue.push(std::make_pair(dists[v], v));
        }
      }
    }
    return dists;
  }

  GraphType A_;
  std::vector<IndexType> labels_;
};

#ifdef _METIS_H_
namespace METIS {

template <typename Graph, typename T>
std::vector<Graph> splitGraph(const Graph &G, const T &part) {
  using ValueType = typename Graph::ValueType;
  using IndexType = typename Graph::IndexType;
  IndexType K = 0;
  for (const auto &it : part) K = K < it ? it : K;
  ++K;
  std::vector<Graph> retval(K);
#pragma omp parallel for
  for (IndexType l = 0; l < K; ++l) {
    IndexType nnodes = 0;
    std::vector<Eigen::Triplet<ValueType, IndexType>> trips;
    trips.reserve(G.graph().nonZeros() / K);
    std::vector<IndexType> index_map(G.nnodes(), -1);
    for (IndexType i = 0; i < G.nnodes(); ++i)
      if (part[i] == l) index_map[i] = nnodes++;
    const IndexType *ia = G.graph().outerIndexPtr();
    const IndexType *ja = G.graph().innerIndexPtr();
    const ValueType *aij = G.graph().valuePtr();
    for (IndexType i = 0; i < G.nnodes(); ++i)
      if (part[i] == l)
        for (IndexType k = ia[i]; k < ia[i + 1]; ++k) {
          const IndexType j = ja[k];
          const ValueType val = aij[k];
          if (part[j] == l)
            trips.push_back(Eigen::Triplet<ValueType, IndexType>(
                index_map[i], index_map[j], val));
        }
    retval[l].init(nnodes, trips);
    for (IndexType i = 0; i < G.nnodes(); ++i)
      if (part[i] == l) retval[l].labels()[index_map[i]] = i;
  }
  return retval;
}

template <typename ValueType>
std::vector<idx_t> partitionGraph(Graph<idx_t, ValueType> &G, Index K) {
  idx_t nvtxs = G.nnodes();
  idx_t ncon = 1;
  idx_t nparts = K;
  idx_t wgtflag = 1;  // only edge weights
  idx_t numflag = 0;
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);

  idx_t objval;
  std::vector<idx_t> part(nvtxs);
  idx_t *xadj = (G.graph()).outerIndexPtr();
  idx_t *adjncy = (G.graph()).innerIndexPtr();
  std::vector<idx_t> adjwgt(G.nedges());
  for (idx_t i = 0; i < adjwgt.size(); ++i)
    adjwgt[i] = 1. / (1e-6 + G.graph().valuePtr()[i] * G.graph().valuePtr()[i]);

  int status = METIS_PartGraphRecursive(
      &nvtxs, &ncon, xadj, adjncy, adjwgt.data(), NULL, NULL, &nparts, NULL,
      NULL, options, &objval, part.data());
  assert(status == METIS_OK);
  return part;
}

template <typename ValueType>
std::vector<idx_t> partitionGraphKWay(Graph<idx_t, ValueType> &G, Index K) {
  idx_t nvtxs = G.nnodes();
  idx_t ncon = 1;
  idx_t nparts = K;
  idx_t wgtflag = 1;  // only edge weights
  idx_t numflag = 0;
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);

  idx_t objval;
  std::vector<idx_t> part(nvtxs);
  idx_t *xadj = (G.graph()).outerIndexPtr();
  idx_t *adjncy = (G.graph()).innerIndexPtr();
  std::vector<idx_t> adjwgt(G.nedges());
  for (idx_t i = 0; i < adjwgt.size(); ++i)
    adjwgt[i] = 1. / (1e-6 + G.graph().valuePtr()[i] * G.graph().valuePtr()[i]);

  int status = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, NULL, NULL,
                                   adjwgt.data(), &nparts, NULL, NULL, options,
                                   &objval, part.data());
  assert(status == METIS_OK);
  return part;
}

}  // namespace METIS
#endif
}  // namespace FMCA
#endif
