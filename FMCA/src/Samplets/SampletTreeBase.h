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
#ifndef FMCA_SAMPLETS_SAMPLETTREEBASE_H_
#define FMCA_SAMPLETS_SAMPLETTREEBASE_H_

namespace FMCA {

/**
 *  \ingroup Samplets
 *  \brief SampletTreeNodeBase defines the basic fields required for an
 *         abstract SampletTree, i.e. the transformation matrices
 **/
template <typename Derived> struct SampletTreeNodeDataFields {
  Matrix Q_;
  Matrix mom_buffer_;
  Index nscalfs_;
  Index nsamplets_;
  Index start_index_;
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
  typedef typename internal::traits<Derived>::Node Node;
  typedef ClusterTreeBase<Derived> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::bb;
  using Base::block_id;
  using Base::derived;
  using Base::indices;
  using Base::indices_begin;
  using Base::init;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
  using Base::dad;
  //////////////////////////////////////////////////////////////////////////////
  void sampletTransformMatrix(Matrix &M) {
    M = sampletTransform(M);
    M = sampletTransform(M.transpose()).transpose();
  }
  //////////////////////////////////////////////////////////////////////////////
  void inverseSampletTransformMatrix(Matrix &M) {
    M = inverseSampletTransform(M);
    M = inverseSampletTransform(M.transpose()).transpose();
  }
  //////////////////////////////////////////////////////////////////////////////
  Matrix sampletTransform(const Matrix &data) const {
    assert(is_root() &&
           "sampletTransform needs to be called from the root cluster");
    Matrix retval(data.rows(), data.cols());
    retval.setZero();
    sampletTransformRecursion(data, &retval);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  Matrix inverseSampletTransform(const Matrix &data) const {
    assert(is_root() &&
           "sampletTransform needs to be called from the root cluster");
    Matrix retval(data.rows(), data.cols());
    retval.setZero();
    inverseSampletTransformRecursion(data, &retval, nullptr);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<Scalar>> transformationMatrixTriplets() const {
    const Index n = indices().size();
    Matrix buffer(n, 1);
    Matrix unit(n, 1);
    std::vector<Eigen::Triplet<Scalar>> triplet_list;
    for (auto j = 0; j < n; ++j) {
      buffer.setZero();
      unit.setZero();
      unit(j) = 1;
      buffer = sampletTransform(unit);
      for (auto i = 0; i < buffer.size(); ++i)
        if (abs(buffer(i)) > 1e-14)
          triplet_list.emplace_back(Eigen::Triplet<Scalar>(i, j, buffer(i)));
    }
    return triplet_list;
  }
  //////////////////////////////////////////////////////////////////////////////
  Index nscalfs() const { return node().nscalfs_; }
  Index nsamplets() const { return node().nsamplets_; }

  Index start_index() const { return node().start_index_; }
  //////////////////////////////////////////////////////////////////////////////
  const Matrix &Q() const { return node().Q_; }

private:
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  Matrix sampletTransformRecursion(const Matrix &data, Matrix *svec) const {
    Matrix retval(0, 0);
    Index scalf_shift = 0;
    if (is_root())
      scalf_shift = nscalfs();
    if (nSons()) {
      for (auto i = 0; i < nSons(); ++i) {
        Matrix scalf = sons(i).sampletTransformRecursion(data, svec);
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
  void inverseSampletTransformRecursion(const Matrix &data, Matrix *fvec,
                                        Matrix *ddata,
                                        Index ddata_offset = 0) const {
    Matrix retval;
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
