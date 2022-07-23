// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU General Public License version 3
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_SAMPLETS_SAMPLETTREE_H_
#define FMCA_SAMPLETS_SAMPLETTREE_H_

namespace FMCA {

struct SampletTreeNode : public SampletTreeNodeBase<SampletTreeNode> {};

namespace internal {
template <typename ClusterTreeType>
struct traits<SampletTree<ClusterTreeType>> : public traits<ClusterTreeType> {
  typedef SampletTreeNode Node;
};
} // namespace internal

/**
 *  \ingroup Samplets
 *  \brief The SampletTree class manages samplets constructed on a cluster tree.
 */
template <typename ClusterTreeType>
struct SampletTree : public SampletTreeBase<SampletTree<ClusterTreeType>> {
public:
  typedef typename internal::traits<SampletTree>::Node Node;
  typedef SampletTreeBase<SampletTree<ClusterTreeType>> Base;
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
  SampletTree() {}
  template <typename Moments, typename... Ts>
  SampletTree(const Moments &mom, Index min_cluster_size, Ts &&...ts) {
    init(mom, min_cluster_size, std::forward<Ts>(ts)...);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  template <typename Moments, typename... Ts>
  void init(const Moments &mom, Index min_cluster_size, Ts &&...ts) {
    const Index mincsize = min_cluster_size > mom.interp().Xi().cols()
                                   ? min_cluster_size
                                   : mom.interp().Xi().cols();
    internal::ClusterTreeInitializer<ClusterTreeType>::init(
        *this, mincsize, std::forward<Ts>(ts)...);
    computeSamplets(mom);
    internal::sampletMapper<SampletTree>(*this);
    return;
  }

private:
  template <typename Moments> void computeSamplets(const Moments &mom) {
    if (nSons()) {
      Index offset = 0;
      for (auto i = 0; i < nSons(); ++i) {
        sons(i).computeSamplets(mom);
        // the son now has moments, lets grep them...
        Matrix shift = 0.5 * (sons(i).bb().col(0) - bb().col(0) +
                                   sons(i).bb().col(1) - bb().col(1));
        node().mom_buffer_.conservativeResize(
            sons(i).node().mom_buffer_.rows(),
            offset + sons(i).node().mom_buffer_.cols());
        node().mom_buffer_.block(0, offset, sons(i).node().mom_buffer_.rows(),
                                 sons(i).node().mom_buffer_.cols()) =
            mom.shift_matrix(shift) * sons(i).node().mom_buffer_;
        offset += sons(i).node().mom_buffer_.cols();
        // clear moment buffer of the children
        sons(i).node().mom_buffer_.resize(0, 0);
      }
    } else
      // compute cluster basis of the leaf
      node().mom_buffer_ = mom.moment_matrix(*this);
    // are there samplets?
    if (mom.mdtilde() < node().mom_buffer_.cols()) {
      Eigen::HouseholderQR<Matrix> qr(node().mom_buffer_.transpose());
      node().Q_ = qr.householderQ();
      node().nscalfs_ = mom.mdtilde();
      node().nsamplets_ = node().Q_.cols() - node().nscalfs_;
      // this is the moment for the dad cluster
      node().mom_buffer_ = qr.matrixQR()
                               .block(0, 0, mom.mdtilde(), mom.mdtilde2())
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
} // namespace FMCA
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
