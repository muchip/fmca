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
#ifndef FMCA_SAMPLETS_SAMPLETTREEBASE_H_
#define FMCA_SAMPLETS_SAMPLETTREEBASE_H_

namespace FMCA {

/**
 *  \ingroup Samplets
 *  \brief SampletTreeNodeBase defines the basic fields required for an
 *         abstract SampletTree, i.e. the transformation matrices
 **/
template <typename Derived> struct SampletTreeNodeDataFields {
  typename internal::traits<Derived>::eigenMatrix Q_;
  typename internal::traits<Derived>::eigenMatrix mom_buffer_;
  IndexType nscalfs_;
  IndexType nsamplets_;
  IndexType start_index_;
  IndexType block_id_;
};

template <typename Derived>
struct SampletTreeNodeBase : public ClusterTreeNodeBase<Derived>,
                             public SampletTreeNodeDataFields<Derived> {};
/**
 *  \ingroup Samplets
 *  \brief The SampletTreeBase class manages abstract samplet trees
 **/
template <typename Derived>
struct SampletTreeBase : public ClusterTreeBase<Derived> {
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  typedef typename internal::traits<Derived>::node_type node_type;
  typedef typename internal::traits<Derived>::value_type value_type;
  typedef ClusterTreeBase<Derived> Base;
  // make base class methods visible
  using TreeBase<Derived>::is_root;
  using Base::appendSons;
  using Base::indices;
  using Base::indices_begin;
  using Base::init;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;

  //////////////////////////////////////////////////////////////////////////////
  void sampletTransformMatrix(eigenMatrix &M) {
    M = sampletTransform(M);
    M = sampletTransform(M.transpose()).transpose();
  }
  //////////////////////////////////////////////////////////////////////////////
  void inverseSampletTransformMatrix(eigenMatrix &M) {
    M = inverseSampletTransform(M);
    M = inverseSampletTransform(M.transpose()).transpose();
  }
  //////////////////////////////////////////////////////////////////////////////
  eigenMatrix sampletTransform(const eigenMatrix &data) const {
    assert(is_root() &&
           "sampletTransform needs to be called from the root cluster");
    eigenMatrix retval(data.rows(), data.cols());
    retval.setZero();
    sampletTransformRecursion(data, &retval);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  eigenMatrix inverseSampletTransform(const eigenMatrix &data) const {
    assert(is_root() &&
           "sampletTransform needs to be called from the root cluster");
    eigenMatrix retval(data.rows(), data.cols());
    retval.setZero();
    inverseSampletTransformRecursion(data, &retval, nullptr);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<value_type>> transformationMatrixTriplets() const {
    const IndexType n = indices().size();
    eigenMatrix buffer(n, 1);
    std::vector<Eigen::Triplet<value_type>> triplet_list;
    for (auto j = 0; j < n; ++j) {
      buffer.setZero();
      buffer = sampletTransform(eigenMatrix::Unit(n, j));
      for (auto i = 0; i < buffer.size(); ++i)
        if (abs(buffer(i)) > 1e-14)
          triplet_list.emplace_back(
              Eigen::Triplet<value_type>(i, j, buffer(i)));
    }
    return triplet_list;
  }
  //////////////////////////////////////////////////////////////////////////////
  IndexType nscalfs() const { return node().nscalfs_; }
  IndexType nsamplets() const { return node().nsamplets_; }
  IndexType block_id() const { return node().block_id_; }
  IndexType start_index() const { return node().start_index_; }
  //////////////////////////////////////////////////////////////////////////////
  const eigenMatrix &Q() const { return node().Q_; }

protected:
  //////////////////////////////////////////////////////////////////////////////
  void sampletMapper() {
    assert(is_root() &&
           "sampletMapper needs to be called from the root cluster");
    IndexType i = 0;
    IndexType sum = 0;
    // assign vector start_index to each wavelet cluster
    for (auto &it : *this) {
      it.node().start_index_ = sum;
      it.node().block_id_ = i;
      sum += it.derived().nsamplets();
      if (it.is_root())
        sum += it.derived().nscalfs();
    }
    assert(sum == indices().size());
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  eigenMatrix sampletTransformRecursion(const eigenMatrix &data,
                                        eigenMatrix *svec) const {
    eigenMatrix retval(0, 0);
    IndexType scalf_shift = 0;
    if (is_root())
      scalf_shift = nscalfs();
    if (nSons()) {
      for (auto i = 0; i < nSons(); ++i) {
        eigenMatrix scalf = sons(i).sampletTransformRecursion(data, svec);
        retval.conservativeResize(retval.rows() + scalf.rows(), data.cols());
        retval.bottomRows(scalf.rows()) = scalf;
      }
    } else {
      retval = data.middleRows(indices_begin(), indices().size());
    }
    if (nsamplets()) {
      svec->middleRows(start_index() + scalf_shift, nsamplets()) =
          Q().rightCols(nsamplets()).transpose() * retval;
      retval = Q().leftCols(nscalfs()).transpose() * retval;
    }
    if (is_root())
      svec->middleRows(start_index(), nscalfs()) = retval;
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  void inverseSampletTransformRecursion(const eigenMatrix &data,
                                        eigenMatrix *fvec, eigenMatrix *ddata,
                                        IndexType ddata_offset = 0) const {
    eigenMatrix retval;
    if (is_root()) {
      retval =
          Q().leftCols(nscalfs()) * data.middleRows(start_index(), nscalfs()) +
          Q().rightCols(nsamplets()) *
              data.middleRows(start_index() + nscalfs(), nsamplets());
    } else {
      retval =
          Q().leftCols(nscalfs()) * ddata->middleRows(ddata_offset, nscalfs());
      if (nsamplets())
        retval += Q().rightCols(nsamplets()) *
                  data.middleRows(start_index(), nsamplets());
    }
    if (nSons()) {
      ddata_offset = 0;
      for (auto i = 0; i < nSons(); ++i) {
        sons(i).inverseSampletTransformRecursion(data, fvec, &retval,
                                                 ddata_offset);
        ddata_offset += sons(i).nscalfs();
      }
    } else {
      fvec->middleRows(indices_begin(), retval.rows()) = retval;
    }
    return;
  }
};
} // namespace FMCA
#endif
