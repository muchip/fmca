// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifdef _METIS_H_
#ifndef FMCA_SAMPLETS_GRAPHSAMPLETFOREST_H_
#define FMCA_SAMPLETS_GRAPHSAMPLETFOREST_H_

namespace FMCA {
template <typename Graph>
class GraphSampletForest {
 public:
  using ValueType = typename Graph::ValueType;
  using IndexType = typename Graph::IndexType;
  using SampletInterpolator = FMCA::MonomialInterpolator;
  using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
  using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;
  GraphSampletForest() {};
  GraphSampletForest(Graph &G, const Index M, const Index emb_dim,
                     const Index dtilde, const Index nlm = 100) {
    init(G, M, emb_dim, dtilde, nlm);
    return;
  }
  void init(Graph &G, const Index M, const Index emb_dim, const Index dtilde,
            const Index nlm = 100) {
    M_ = M;
    dtilde_ = dtilde > 0 ? dtilde : 1;
    parts_ = METIS::partitionGraphKWay(G, M_);
    sub_graphs_ = METIS::splitGraph(G, parts_);
    global_ids_.resize(M_);
    points_.resize(M_);
    trees_.resize(M_);
    nrgs_.resize(M_);
    global_pos_.resize(M_ + 1);
#pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < M_; ++i) {
      const IndexType n =
          nlm > sub_graphs_[i].nnodes() ? sub_graphs_[i].nnodes() : nlm;
      points_[i] = LIsomap(sub_graphs_[i], n, emb_dim, nrgs_.data() + i);
      const SampletMoments samp_mom(points_[i], dtilde_ - 1);
      trees_[i].init(samp_mom, 0, points_[i]);
      global_ids_[i].resize(trees_[i].block_size());
      for (Index j = 0; j < global_ids_.size(); ++j)
        global_ids_[i][j] = sub_graphs_[i].labels()[trees_[i].indices()[j]];
    }
    global_pos_[0] = 0;
    for (Index i = 1; i <= M_; ++i)
      global_pos_[i] = global_pos_[i - 1] + global_ids_[i - 1].size();
    return;
  }
  const std::vector<Scalar> &lost_energies() { return nrgs_; }
  Matrix sampletTransform(const Matrix &lhs) {
    Matrix retval(lhs.rows(), lhs.cols());
    for (Index i = 0; i < M_; ++i) {
      Matrix loc_rhs(global_ids_[i].size(), lhs.cols());
      for (Index j = 0; j < global_ids_[i].size(); ++j)
        loc_rhs.row(j) = lhs.row(global_ids_[i][j]);
      loc_rhs = trees_[i].sampletTransform(loc_rhs);
      retval.middleRows(global_pos_[i], global_ids_[i].size()) = loc_rhs;
    }
    return retval;
  }

  Matrix inverseSampletTransform(const Matrix &lhs) {
    Matrix retval(lhs.rows(), lhs.cols());
    for (Index i = 0; i < M_; ++i) {
      Matrix loc_rhs = trees_[i].inverseSampletTransform(
          lhs.middleRows(global_pos_[i], global_ids_[i].size()));
      for (Index j = 0; j < global_ids_[i].size(); ++j)
        retval.row(global_ids_[i][j]) = loc_rhs.row(j);
    }
    return retval;
  }

 private:
  std::vector<Graph> sub_graphs_;
  std::vector<SampletTree> trees_;
  std::vector<Matrix> points_;
  std::vector<IndexType> parts_;
  std::vector<IndexType> global_pos_;
  std::vector<std::vector<IndexType>> global_ids_;
  std::vector<Scalar> nrgs_;
  IndexType M_;
  Index dtilde_;
};
}  // namespace FMCA
#endif
#endif
