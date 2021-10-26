// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_SAMPLETS_H2SAMPLETTREE_H_
#define FMCA_SAMPLETS_H2SAMPLETTREE_H_

namespace FMCA {

namespace internal {
template <> struct traits<H2SampletTreeNode> {
  typedef FloatType value_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  typedef TensorProductInterpolator<value_type> Interpolator;
};
} // namespace internal

struct H2SampletTreeNode : public H2SampletTreeNodeBase<H2SampletTreeNode> {};

namespace internal {
template <> struct traits<H2SampletTree> : public traits<SampletTreeQR> {
  typedef H2SampletTreeNode node_type;
  typedef traits<H2SampletTreeNode>::Interpolator Interpolator;
};
} // namespace internal

/**
 *  \ingroup Samplets
 *  \brief The SampletTree class manages samplets constructed on a cluster tree.
 */
struct H2SampletTree : public H2SampletTreeBase<H2SampletTree> {
public:
  typedef typename internal::traits<H2SampletTree>::value_type value_type;
  typedef typename internal::traits<H2SampletTree>::node_type node_type;
  typedef typename internal::traits<H2SampletTree>::eigenMatrix eigenMatrix;
  typedef H2SampletTreeBase<H2SampletTree> Base;
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
      pXi_ = std::make_shared<eigenMatrix>();
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
