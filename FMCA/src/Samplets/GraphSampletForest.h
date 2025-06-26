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
    }
    global_pos_[0] = 0;
    for (Index i = 1; i <= M_; ++i)
      global_pos_[i] = global_pos_[i - 1] + sub_graphs_[i - 1].labels().size();
    return;
  }
  const std::vector<Scalar> &lost_energies() { return nrgs_; }

  Matrix sampletTransform(const Matrix &lhs, const Scalar thres = 0) {
    Matrix retval(lhs.rows(), lhs.cols());
#pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < M_; ++i) {
      Matrix loc_rhs = global2local(lhs, i);
      loc_rhs = trees_[i].toClusterOrder(loc_rhs);
      loc_rhs = trees_[i].sampletTransform(loc_rhs);
      if (thres > 0) {
        Vector nrms = loc_rhs.colwise().norm();
        for (Index j = 0; j < loc_rhs.cols(); ++j) {
          loc_rhs.col(j) = (loc_rhs.col(j).array().abs() >= thres * nrms(j))
                               .select(loc_rhs.col(j), 0.0);
        }
      }
      retval.middleRows(global_pos_[i], sub_graphs_[i].labels().size()) =
          loc_rhs;
    }
    return retval;
  }

  Matrix inverseSampletTransform(const Matrix &lhs) {
    Matrix retval(lhs.rows(), lhs.cols());
#pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < M_; ++i) {
      Matrix loc_lhs =
          lhs.middleRows(global_pos_[i], sub_graphs_[i].labels().size());
      loc_lhs = trees_[i].inverseSampletTransform(loc_lhs);
      loc_lhs = trees_[i].toNaturalOrder(loc_lhs);
      local2global(retval, loc_lhs, i);
    }
    return retval;
  }

 private:
  Matrix global2local(const Matrix &rhs, const Index i) {
    Matrix loc_rhs(sub_graphs_[i].labels().size(), rhs.cols());
    for (Index j = 0; j < sub_graphs_[i].labels().size(); ++j)
      loc_rhs.row(j) = rhs.row(sub_graphs_[i].labels()[j]);
    return loc_rhs;
  }
  void local2global(Matrix &lhs, const Matrix &loc_lhs, Index i) {
    for (Index j = 0; j < sub_graphs_[i].labels().size(); ++j)
      lhs.row(sub_graphs_[i].labels()[j]) = loc_lhs.row(j);
    return;
  }
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
