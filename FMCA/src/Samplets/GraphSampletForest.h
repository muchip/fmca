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
    parts_ = std::vector<IndexType>(G.nnodes(), 0);
    if (M_ > 1) parts_ = METIS::partitionGraphKWay(G, M_);
    sub_graphs_ = METIS::splitGraph(G, parts_);
    global_ids_.resize(M_);
    lm_ids_.resize(M_);
    points_.resize(M_);
    trees_.resize(M_);
    nrgs_.resize(M_);
    global_pos_.resize(M_ + 1);
#pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < M_; ++i) {
      const IndexType n =
          nlm > sub_graphs_[i].nnodes() ? sub_graphs_[i].nnodes() : nlm;
      points_[i] = LIsomap(sub_graphs_[i], n, emb_dim, nrgs_.data() + i,
                           std::addressof(lm_ids_[i]));
    }
    global_pos_[0] = 0;
    for (Index i = 1; i <= M_; ++i)
      global_pos_[i] = global_pos_[i - 1] + sub_graphs_[i - 1].nnodes();

    init_sampletTrees(dtilde);
    return;
  }

  void init_sampletTrees(const Index dtilde) {
    dtilde_ = dtilde > 0 ? dtilde : 1;
    trees_.clear();
    trees_.resize(M_);
#pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < M_; ++i) {
      const SampletMoments samp_mom(points_[i], dtilde_ - 1);
      trees_[i].init(samp_mom, 0, points_[i]);
    }
    return;
  }

  const std::vector<Scalar> &lost_energies() const { return nrgs_; }
  const std::vector<IndexType> &parts() const { return parts_; }
  const std::vector<Graph> &sub_graphs() const { return sub_graphs_; }
  const std::vector<SampletTree> &samplet_trees() const { return trees_; }
  const std::vector<Matrix> &points() const { return points_; }
  const std::vector<std::vector<IndexType>> &lm_ids() const { return lm_ids_; }

  Matrix sampletTransform(const Matrix &lhs) const {
    Matrix retval(lhs.rows(), lhs.cols());
#pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < M_; ++i) {
      Matrix loc_rhs = global2local(lhs, i);
      loc_rhs = trees_[i].toClusterOrder(loc_rhs);
      loc_rhs = trees_[i].sampletTransform(loc_rhs);
      retval.middleRows(global_pos_[i], sub_graphs_[i].nnodes()) = loc_rhs;
    }
    return retval;
  }

  Matrix inverseSampletTransform(const Matrix &lhs) {
    Matrix retval(lhs.rows(), lhs.cols());
#pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < M_; ++i) {
      Matrix loc_lhs = lhs.middleRows(global_pos_[i], sub_graphs_[i].nnodes());
      loc_lhs = trees_[i].inverseSampletTransform(loc_lhs);
      loc_lhs = trees_[i].toNaturalOrder(loc_lhs);
      local2global(retval, loc_lhs, i);
    }
    return retval;
  }

  Vector threshold(const Vector &lhs, const Scalar thres) {
    Vector retval(lhs.size());
    for (Index i = 0; i < M_; ++i) {
      const Vector loc_lhs =
          lhs.segment(global_pos_[i], sub_graphs_[i].nnodes());
      const Scalar norm2 = loc_lhs.squaredNorm();
      std::vector<const SampletTree *> adaptive_tree =
          adaptiveTreeSearch<SampletTree>(trees_[i], loc_lhs, thres * norm2);

      Vector thres_tdata = loc_lhs;
      Index nnz = 0;
      {
        thres_tdata.setZero();

        for (Index j = 0; j < adaptive_tree.size(); ++j) {
          if (adaptive_tree[j] != nullptr) {
            const SampletTree &node = *(adaptive_tree[j]);
            const Index ndist =
                node.is_root() ? node.Q().cols() : node.nsamplets();
            thres_tdata.segment(node.start_index(), ndist) =
                loc_lhs.segment(node.start_index(), ndist);
            nnz += ndist;
          }
        }
      }
      retval.segment(global_pos_[i], sub_graphs_[i].nnodes()) = thres_tdata;
    }
    return retval;
  }

 private:
  Matrix global2local(const Matrix &rhs, const Index i) const {
    Matrix loc_rhs(sub_graphs_[i].nnodes(), rhs.cols());
    for (Index j = 0; j < sub_graphs_[i].nnodes(); ++j)
      loc_rhs.row(j) = rhs.row(sub_graphs_[i].labels()[j]);
    return loc_rhs;
  }
  void local2global(Matrix &lhs, const Matrix &loc_lhs, Index i) const {
    for (Index j = 0; j < sub_graphs_[i].nnodes(); ++j)
      lhs.row(sub_graphs_[i].labels()[j]) = loc_lhs.row(j);
    return;
  }
  std::vector<Graph> sub_graphs_;
  std::vector<SampletTree> trees_;
  std::vector<Matrix> points_;
  std::vector<IndexType> parts_;
  std::vector<IndexType> global_pos_;
  std::vector<std::vector<IndexType>> global_ids_;
  std::vector<std::vector<IndexType>> lm_ids_;
  std::vector<Scalar> nrgs_;
  IndexType M_;
  Index dtilde_;
};
}  // namespace FMCA
#endif
#endif
