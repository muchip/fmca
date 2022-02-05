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
template <>
struct traits<H2SampletTreeNode> {
  typedef FloatType value_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  typedef TotalDegreeInterpolator<value_type> Interpolator;
};
}  // namespace internal

struct H2SampletTreeNode : public H2SampletTreeNodeBase<H2SampletTreeNode> {};

namespace internal {
template <>
struct traits<H2SampletTree> : public traits<SampletTreeQR> {
  typedef H2SampletTreeNode node_type;
  typedef traits<H2SampletTreeNode>::Interpolator Interpolator;
};
}  // namespace internal

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
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  H2SampletTree() {}
  H2SampletTree(const eigenMatrix &P, IndexType min_cluster_size = 1,
                IndexType dtilde = 1, IndexType polynomial_degree = 3) {
    init(P, min_cluster_size, dtilde, polynomial_degree);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  void init(const eigenMatrix &P, IndexType min_cluster_size = 1,
            IndexType dtilde = 1, IndexType polynomial_degree = 3) {
    // init moment computer
    Base::init(P, min_cluster_size, polynomial_degree);
    SampleMomentComputer<H2SampletTree, MultiIndexSet<TotalDegree>> mom_comp;
    mom_comp.init(P.rows(), dtilde);
    computeSamplets(P, mom_comp);

    computeMultiscaleClusterBasis();
    updateMultiscaleClusterBasis();
    sampletMapper();
    return;
  }

 private:
  template <typename MomentComputer>
  void computeSamplets(const eigenMatrix &P, const MomentComputer &mom_comp) {
    if (nSons()) {
      IndexType offset = 0;
      for (auto i = 0; i < nSons(); ++i) {
        sons(i).computeSamplets(P, mom_comp);
        // the son now has moments, lets grep them...
        eigenMatrix shift = 0.5 * (sons(i).bb().col(0) - bb().col(0) +
                                   sons(i).bb().col(1) - bb().col(1));
        node().mom_buffer_.conservativeResize(
            sons(i).node().mom_buffer_.rows(),
            offset + sons(i).node().mom_buffer_.cols());
        node().mom_buffer_.block(0, offset, sons(i).node().mom_buffer_.rows(),
                                 sons(i).node().mom_buffer_.cols()) =
            mom_comp.shift_matrix(shift) * sons(i).node().mom_buffer_;
        offset += sons(i).node().mom_buffer_.cols();
        // clear moment buffer of the children
        sons(i).node().mom_buffer_.resize(0, 0);
      }
    } else
      // compute cluster basis of the leaf
      node().mom_buffer_ = mom_comp.moment_matrix(P, *this);
    // are there samplets?
    if (mom_comp.mdtilde() < node().mom_buffer_.cols()) {
      Eigen::HouseholderQR<eigenMatrix> qr(node().mom_buffer_.transpose());
      node().Q_ = qr.householderQ();
      node().nscalfs_ = mom_comp.mdtilde();
      node().nsamplets_ = node().Q_.cols() - node().nscalfs_;
      // this is the moment for the dad cluster
      node().mom_buffer_ =
          qr.matrixQR()
              .block(0, 0, mom_comp.mdtilde(), mom_comp.mdtilde2())
              .template triangularView<Eigen::Upper>()
              .transpose();
    } else {
      node().Q_ = eigenMatrix::Identity(node().mom_buffer_.cols(),
                                        node().mom_buffer_.cols());
      node().nscalfs_ = node().mom_buffer_.cols();
      node().nsamplets_ = 0;
    }
    return;
  }
};
}  // namespace FMCA
#endif
