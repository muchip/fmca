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
#ifndef FMCA_HMATRIX_HMATRIX_H_
#define FMCA_HMATRIX_HMATRIX_H_

namespace FMCA {

template <typename ClusterTreeType>
struct HMatrixNode : public HMatrixNodeBase<HMatrixNode<ClusterTreeType>> {};

namespace internal {

template <typename ClusterTreeType>
struct traits<HMatrixNode<ClusterTreeType>> : public traits<ClusterTreeType> {
  typedef ClusterTreeType RowCType;
  typedef ClusterTreeType ColCType;
};

template <typename ClusterTreeType, typename ClusterComparison>
struct traits<HMatrix<ClusterTreeType, ClusterComparison>>
    : public traits<ClusterTreeType> {
  typedef HMatrixNode<ClusterTreeType> Node;
  typedef typename traits<Node>::RowCType RowCType;
  typedef typename traits<Node>::ColCType ColCType;
};
}  // namespace internal

/**
 *  \ingroup HMatrix
 *  \brief The HMatrix class manages H2 matrices for a given
 *         ClusterTree.

 */
template <typename Derived, typename ClusterComparison = CompareCluster>
struct HMatrix : public HMatrixBase<HMatrix<Derived, ClusterComparison>> {
  typedef HMatrixBase<HMatrix<Derived, ClusterComparison>> Base;
  typedef ClusterComparison CC;
  // make base class methods visible
  using Base::computeHMatrix;
  using Base::node;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  HMatrix() {}
  template <typename EntryGenerator>
  HMatrix(const H2ClusterTreeBase<Derived> &CT, const EntryGenerator &e_gen,
          Scalar eta = 0.8) {
    init(CT, e_gen, eta);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  template <typename MatrixEvaluator>
  void init(const H2ClusterTreeBase<Derived> &CT,
            const MatrixEvaluator &mat_eval, Scalar eta = 0.8) {
    computeHMatrix(CT.derived(), CT.derived(), mat_eval, eta);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
};

}  // namespace FMCA
#endif
