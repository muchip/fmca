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
#include <queue>
//
#include <Eigen/Sparse>
//
#include "Macros.h"
#include "graphIO.h"

namespace FMCA {

template <typename IndexType, typename ValueType>
class Graph {
 public:
  using GraphType = Eigen::SparseMatrix<ValueType, Eigen::RowMajor, IndexType>;
  Graph() {};
  template <typename T>
  void init(const Index nnodes, const T &triplets) {
    A_.resize(nnodes, nnodes);
    labels_.resize(nnodes);
    std::iota(labels_.begin(), labels_.end(), 0);
    A_.setFromTriplets(triplets.begin(), triplets.end());
    A_.makeCompressed();
    return;
  }

  Graph split(const std::vector<IndexType> &part) {
    Graph retval;
    std::vector<IndexType> labels0(part.size(), 0);
    std::vector<IndexType> labels1(part.size(), 0);
    std::vector<Eigen::Triplet<ValueType, IndexType>> trips0;
    std::vector<Eigen::Triplet<ValueType, IndexType>> trips1;
    std::vector<IndexType> index_map(part.size(), 0);
    {
      IndexType k = 0;
      IndexType l = 0;
      for (IndexType i = 0; i < part.size(); ++i) {
        if (0 == part[i]) {
          labels0[k] = labels_[i];
          index_map[i] = k++;
        } else if (1 == part[i]) {
          labels1[l] = labels_[i];
          index_map[i] = l++;
        }
      }
      labels0.resize(k);
      labels1.resize(l);
    }

    {
      trips0.reserve(labels0.size());
      trips1.reserve(labels1.size());
      const IndexType *ia = A_.outerIndexPtr();
      const IndexType *ja = A_.innerIndexPtr();
      const ValueType *aij = A_.valuePtr();
      for (IndexType i = 0; i < A_.rows(); ++i)
        for (IndexType k = ia[i]; k < ia[i + 1]; ++k) {
          const IndexType j = ja[k];
          const ValueType val = aij[k];
          if (0 == part[i] && part[i] == part[j])
            trips0.push_back(Eigen::Triplet<ValueType, IndexType>(
                index_map[i], index_map[j], val));
          else if (1 == part[i] && part[i] == part[j])
            trips1.push_back(Eigen::Triplet<ValueType, IndexType>(
                index_map[i], index_map[j], val));
        }
    }
    A_.resize(labels0.size(), labels0.size());
    A_.setFromTriplets(trips0.begin(), trips0.end());
    A_.makeCompressed();
    labels_ = labels0;
    retval.A_.resize(labels1.size(), labels1.size());
    retval.A_.setFromTriplets(trips1.begin(), trips1.end());
    retval.A_.makeCompressed();
    retval.labels_ = labels1;
    return retval;
  }

  const IndexType nnodes() const { return A_.rows(); }
  const IndexType nedges() const { return A_.nonZeros(); }
  const std::vector<IndexType> &labels() const { return labels_; }
  GraphType &graph() { return A_; }
  const GraphType &graph() const { return A_; }
  Matrix distanceMatrix() const { return FloydWarshall(); }

  template <typename T>
  std::vector<std::vector<ValueType>> partialDistanceMatrix(
      const T &indices) const {
    return Dijkstra(indices);
  }
  template <typename Derived>
  void print(const std::string &fileName, const Eigen::MatrixBase<Derived> &P) {
    Derived Ploc(P.rows(), labels_.size());
    for (IndexType i = 0; i < Ploc.cols(); ++i) Ploc.col(i) = P.col(labels_[i]);
    IO::plotGraph(fileName, Ploc, A_);
    return;
  }

  template <typename Derived>
  void printSignal(const std::string &fileName,
                   const Eigen::MatrixBase<Derived> &P, const Vector &sig) {
    IO::plotGraphSignal(fileName, P, A_, sig);
    return;
  }

 private:
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

  template <typename T>
  std::vector<std::vector<ValueType>> Dijkstra(const T &indices) const {
    using qPair = std::pair<ValueType, IndexType>;
    using pQueue =
        std::priority_queue<qPair, std::vector<qPair>, std::greater<qPair>>;
    std::vector<pQueue> pqueues(indices.size());
    std::vector<std::vector<ValueType>> dists(indices.size());
    // initialize distance vectors and priority queues. We opt here for random
    // access at the expense of an indices.size() x nnodes() storage
#pragma omp parallel for
    for (Index i = 0; i < indices.size(); ++i) {
      std::vector<bool> settled(nnodes(), false);
      dists[i].assign(nnodes(), std::numeric_limits<ValueType>::infinity());
      dists[i][indices[i]] = 0;
      pqueues[i].push(std::make_pair(0., indices[i]));
      while (not pqueues[i].empty()) {
        const ValueType du = pqueues[i].top().first;
        const IndexType u = pqueues[i].top().second;
        pqueues[i].pop();
        if (settled[u]) continue;
        settled[u] = true;
        for (typename GraphType::InnerIterator it(A_, u); it; ++it) {
          const IndexType v = it.col();
          const ValueType w = it.value();
          if (dists[i][v] > du + w) {
            dists[i][v] = du + w;
            pqueues[i].push(std::make_pair(dists[i][v], v));
          }
        }
      }
    }
    return dists;
  }

  GraphType A_;
  std::vector<IndexType> labels_;
};

#ifdef _METIS_H_
template <typename ValueType>
std::vector<idx_t> partitionGraph(Graph<idx_t, ValueType> &G) {
  idx_t nvtxs = G.nnodes();
  idx_t ncon = 1;
  idx_t nparts = 2;
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
    adjwgt[i] = 1. / (1e-3 + G.graph().valuePtr()[i] * G.graph().valuePtr()[i]);

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
    adjwgt[i] = 1. / (1e-3 + G.graph().valuePtr()[i] * G.graph().valuePtr()[i]);

  int status = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, NULL, NULL,
                                   adjwgt.data(), &nparts, NULL, NULL, options,
                                   &objval, part.data());
  assert(status == METIS_OK);
  return part;
}

#endif

}  // namespace FMCA
#endif
