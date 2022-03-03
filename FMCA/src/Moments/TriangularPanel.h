// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_MOMENTS_TRIANGULARPANEL_H_
#define FMCA_MOMENTS_TRIANGULARPANEL_H_

namespace FMCA {
/**
 *  \brief realizes a triangular panel with the local coordinates
 *         from the book of Steinbach/Rjasanow for the use in the
 *         semi-analytic quadrature
 **/
struct TriangularPanel {
  template <typename Derived1, typename Derived2, typename Derived3>
  TriangularPanel(const Eigen::MatrixBase<Derived1> &pt1,
                  const Eigen::MatrixBase<Derived2> &pt2,
                  const Eigen::MatrixBase<Derived3> &pt3) {
    init(pt1, pt2, pt3);
  }
  template <typename Derived1, typename Derived2, typename Derived3>
  void init(const Eigen::MatrixBase<Derived1> &pt1,
            const Eigen::MatrixBase<Derived2> &pt2,
            const Eigen::MatrixBase<Derived3> &pt3) {
    // compute element mapping
    affmap_.col(0) = pt1;
    affmap_.col(1) = pt2 - pt1;
    affmap_.col(2) = pt3 - pt1;
    // compute midpoint;
    mp_ = 1. / 3. * (pt1 + pt2 + pt3);
    // determine radius
    radius_ = (pt1 - mp_).norm();
    radius_ = radius_ >= (pt2 - mp_).norm() ? radius_ : (pt2 - mp_).norm();
    radius_ = radius_ >= (pt3 - mp_).norm() ? radius_ : (pt3 - mp_).norm();
    // determine normal
    cs_.col(2) = affmap_.col(1).cross(affmap_.col(2));
    volel_ = cs_.col(2).norm();
    cs_.col(2) /= volel_;
    // determine direction of the opposide
    cs_.col(1) = (pt3 - pt2) / (pt3 - pt2).norm();
    // lot
    cs_.col(0) = cs_.col(1).cross(cs_.col(2));
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix3d cs_;
  Eigen::Matrix3d affmap_;
  Eigen::Vector3d mp_;
  double radius_;
  double volel_;
};

}  // namespace FMCA
#endif
