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
#ifndef FMCA_SAMPLETS_GRAPHSAMPLETTREE_H_
#define FMCA_SAMPLETS_GRAPHSAMPLETTREE_H_

#include "../util/Graph.h"
#include "../util/MDS.h"
namespace FMCA {

struct GraphSampletTreeNode : public SampletTreeNodeBase<GraphSampletTreeNode> {
  Matrix D;
  Matrix P;
};

namespace internal {
template <>
struct traits<GraphSampletTree> : public traits<MetisClusterTree> {
  typedef GraphSampletTreeNode Node;
};
}  // namespace internal

/**
 *  \ingroup Samplets
 *  \brief The SampletTree class manages samplets constructed on a cluster tree.
 */
struct GraphSampletTree : public SampletTreeBase<GraphSampletTree> {
 public:
  typedef typename internal::traits<GraphSampletTree>::Node Node;
  typedef SampletTreeBase<GraphSampletTree> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::bb;
  using Base::block_id;
  using Base::derived;
  using Base::indices;
  using Base::indices_begin;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nsamplets;
  using Base::nscalfs;
  using Base::nSons;
  using Base::Q;
  using Base::sons;
  using Base::start_index;

  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  GraphSampletTree() {}
  template <typename Graph, typename Interpolator>
  GraphSampletTree(const Graph &G, Index emb_dim, Index min_cluster_size,
                   Index dtilde) {
    init<Graph, Interpolator>(G, emb_dim, min_cluster_size, dtilde);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  template <typename Interpolator, typename Graph>
  void init(const Graph &G, Index emb_dim, Index min_cluster_size,
            Index dtilde) {
    const Index q = dtilde > 0 ? dtilde - 1 : 0;
    const Index internal_q = q;  // SampletHelper::internal_q(q, emb_dim);
    Interpolator interp;
    interp.init(emb_dim, internal_q);
    const Index mincsize = min_cluster_size > interp.Xi().cols()
                               ? min_cluster_size
                               : interp.Xi().cols();
    MetisClusterTree::initializer::init(*this, mincsize, G);
    computeSamplets(interp, G);
    internal::sampletMapper<GraphSampletTree>(*this);
    return;
  }

 private:
  template <typename Interpolator, typename Graph>
  void computeSamplets(const Interpolator &interp, const Graph &G) {
    const Index nmom = interp.idcs().index_set().size();
    if (nSons()) {
      Index offset = 0;
      for (auto i = 0; i < nSons(); ++i) {
        sons(i).computeSamplets(interp, G);
        // the son now has moments, lets grep them...
        node().mom_buffer_.conservativeResize(
            sons(i).node().mom_buffer_.rows(),
            offset + sons(i).node().mom_buffer_.cols());
        node().mom_buffer_.block(0, offset, sons(i).node().mom_buffer_.rows(),
                                 sons(i).node().mom_buffer_.cols()) =
            sons(i).node().mom_buffer_;
        offset += sons(i).node().mom_buffer_.cols();
        // clear moment buffer of the children
        sons(i).node().mom_buffer_.resize(0, 0);
      }
    } else {
      // compute cluster basis of the leaf
      std::vector<Eigen::Triplet<FMCA::Scalar>> trips;
      trips.reserve(block_size() * block_size());
      for (FMCA::Index i = 0; i < block_size(); ++i)
        for (FMCA::Index j = 0; j < i; ++j) {
          const FMCA::Scalar w = G.graph().coeff(indices()[i], indices()[j]);
          if (std::abs(w) > FMCA_ZERO_TOLERANCE) {
            trips.push_back(Eigen::Triplet<FMCA::Scalar>(i, j, w));
            trips.push_back(Eigen::Triplet<FMCA::Scalar>(j, i, w));
          }
        }
      Graph G2;
      G2.init(block_size(), trips);
      node().D = G2.distanceMatrix();
      node().P = MDS(node().D, interp.dim());
      if (node().P.rows() < interp.dim()) {
        Matrix Pdim(interp.dim(), node().P.cols());
        Pdim.setZero();
        Pdim.topRows(node().P.rows()) = node().P;
        node().P = Pdim;
      }
      Vector mp = node().P.rowwise().mean();
      node().mom_buffer_.resize(interp.Xi().cols(), node().P.cols());
      for (auto i = 0; i < block_size(); ++i)
        node().mom_buffer_.col(i) =
            interp.evalPolynomials(node().P.col(i) - mp);
    }
    // are there samplets?
    if (nmom < node().mom_buffer_.cols()) {
      Eigen::HouseholderQR<Matrix> qr(node().mom_buffer_.transpose());
      node().Q_ = qr.householderQ();
      node().nscalfs_ = nmom;
      node().nsamplets_ = node().Q_.cols() - node().nscalfs_;
      // this is the moment for the dad cluster
      node().mom_buffer_ = qr.matrixQR()
                               .block(0, 0, nmom, qr.matrixQR().cols())
                               .template triangularView<Eigen::Upper>()
                               .transpose();
    } else {
      node().Q_ = Matrix::Identity(node().mom_buffer_.cols(),
                                   node().mom_buffer_.cols());
      node().nscalfs_ = node().mom_buffer_.cols();
      node().nsamplets_ = 0;
    }
    return;
  }
};
}  // namespace FMCA
#endif

#if 0
  //////////////////////////////////////////////////////////////////////////////
  void visualizeCoefficients(const eigenVector &coeffs,
                             const std::string &filename,
                             value_type thresh = 1e-6) {
    std::vector<Eigen::Matrix3d> bbvec;
    std::vector<value_type> cell_values;
    visualizeCoefficientsRecursion(coeffs, bbvec, cell_values, thresh);
    IO::plotBoxes(filename, bbvec, cell_values);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  void visualizeCoefficientsRecursion(const eigenVector &coeffs,
                                      std::vector<Eigen::Matrix3d> &bbvec,
                                      std::vector<value_type> &cval,
                                      value_type thresh) {
    double color = 0;
    if (!wlevel_) {
      color = coeffs.segment(start_index_, nscalfs_ + nsamplets_)
                  .cwiseAbs()
                  .maxCoeff();
    } else {
      if (nsamplets_)
        color = coeffs.segment(start_index_, nsamplets_).cwiseAbs().maxCoeff();
    }
    if (color > thresh) {
      Eigen::Matrix3d bla;
      bla.setZero();
      if (dimension == 2) {
        bla.topRows(2) = cluster_->get_bb();
      }
      // if (dimension == 3) {
      //   bla = cluster_->get_bb();
      // }
      bbvec.push_back(bla);
      cval.push_back(color);
    }
    if (sons_.size()) {
      for (auto i = 0; i < sons_.size(); ++i)
        sons_[i].visualizeCoefficientsRecursion(coeffs, bbvec, cval, thresh);
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename H2ClusterTree>
  void computeMultiscaleClusterBases(const H2ClusterTree &CT) {
    assert(&(CT.get_cluster()) == cluster_);

    if (!wlevel_) {
      // as I do not have a better solution right now, store the interpolation
      // points within the samplet tree
      pXi_ = std::make_shared<Matrix>();
      *pXi_ = CT.get_Xi();
    }
    if (!sons_.size()) {
      V_ = CT.get_V() * Q_;
    } else {
      // compute multiscale cluster bases of sons and update own
      for (auto i = 0; i < sons_.size(); ++i) {
        sons_[i].pXi_ = pXi_;
        sons_[i].computeMultiscaleClusterBases(CT.get_sons()[i]);
      }
      V_.resize(0, 0);
      for (auto i = 0; i < sons_.size(); ++i) {
        V_.conservativeResize(sons_[i].V_.rows(),
                              V_.cols() + sons_[i].nscalfs_);
        V_.rightCols(sons_[i].nscalfs_) =
            CT.get_E()[i] * sons_[i].V_.leftCols(sons_[i].nscalfs_);
      }
      V_ *= Q_;
    }
    return;
  }

#endif
