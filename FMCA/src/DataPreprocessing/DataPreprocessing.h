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
#ifndef FMCA_DATAPREPROCESSING_DATAPREPROCESSING_H_
#define FMCA_DATAPREPROCESSING_DATAPREPROCESSING_H_

namespace FMCA {

inline Matrix rowMean(const Matrix &X) { return X.rowwise().mean(); }

inline Matrix colMean(const Matrix &X) { return X.colwise().mean(); }

inline Matrix rowVariance(const Matrix &X) {
  return (X.cols() > 1 ? (X.cols() / (X.cols() - 1)) : 0) *
         (X - rowMean(X).replicate(1, X.cols()))
             .array()
             .square()
             .rowwise()
             .mean();
}

inline Matrix colVariance(const Matrix &X) {
  return (X.rows() > 1 ? (X.rows() / (X.rows() - 1)) : 0) *
         (X - colMean(X).replicate(X.rows(), 1))
             .array()
             .square()
             .colwise()
             .mean();
}

inline Matrix rowStd(const Matrix &X) { return rowVariance(X).cwiseSqrt(); }

inline Matrix colStd(const Matrix &X) { return colVariance(X).cwiseSqrt(); }

inline Matrix rowStandardize(const Matrix &X) {
  Matrix retval = X;
  // center data
  retval -= rowMean(X).replicate(1, X.cols());
  // scale standard deviation
  retval.array().colwise() *= rowStd(X).col(0).cwiseInverse().array();

  return retval;
}

inline Matrix colStandardize(const Matrix &X) {
  Matrix retval = X;
  // center data
  retval -= colMean(X).replicate(X.rows(), 1);
  // scale standard deviation
  retval.array().rowwise() *= colStd(X).row(0).cwiseInverse().array();

  return retval;
}

class PCA {
 public:
  PCA(const Matrix &X) : data_(X) {}
  void compute() {
    mean_ = rowMean(data_);
    std_ = rowStd(data_);
    svd_.compute(data_ - mean_.replicate(1, data_.cols()),
                 Eigen::ComputeThinU | Eigen::ComputeThinV);
  }

  const Eigen::BDCSVD<Matrix> &svd() { return svd_; }
  const Matrix &mean() { return mean_; }
  const Matrix &std() { return std_; }

 private:
  Eigen::BDCSVD<Matrix> svd_;
  const Matrix &data_;
  Matrix mean_;
  Matrix std_;
};
}  // namespace FMCA
#endif
