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
#ifndef FMCA_H2MATRIX_H2MATRIX_H_
#define FMCA_H2MATRIX_H2MATRIX_H_

namespace FMCA {

template <typename ClusterTreeType>
struct H2MatrixNode : public H2MatrixNodeBase<H2MatrixNode<ClusterTreeType>> {};

namespace internal {

template <typename ClusterTreeType>
struct traits<H2MatrixNode<ClusterTreeType>> : public traits<ClusterTreeType> {
  typedef ClusterTreeType RowCType;
  typedef ClusterTreeType ColCType;
};

template <typename ClusterTreeType>
struct traits<H2Matrix<ClusterTreeType>> : public traits<ClusterTreeType> {
  typedef H2MatrixNode<ClusterTreeType> Node;
  typedef typename traits<Node>::RowCType RowCType;
  typedef typename traits<Node>::ColCType ColCType;
};
}  // namespace internal

/**
 *  \ingroup H2Matrix
 *  \brief The H2Matrix class manages H2 matrices for a given
 *         H2ClusterTree.

 */
template <typename Derived>
struct H2Matrix : public H2MatrixBase<H2Matrix<Derived>> {
  typedef H2MatrixBase<H2Matrix<Derived>> Base;
  // make base class methods visible
  using Base::computeH2Matrix;
  using Base::node;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  H2Matrix() {}
  template <typename EntryGenerator>
  H2Matrix(const H2ClusterTreeBase<Derived> &CT, const EntryGenerator &e_gen,
           Scalar eta = 0.8) {
    init(CT, e_gen, eta);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  template <typename MatrixEvaluator>
  void init(const H2ClusterTreeBase<Derived> &CT,
            const MatrixEvaluator &mat_eval, Scalar eta = 0.8) {
    computeH2Matrix(CT.derived(), CT.derived(), mat_eval, eta);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
};

}  // namespace FMCA
#endif
