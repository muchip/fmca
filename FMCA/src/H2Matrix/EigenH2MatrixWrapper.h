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
#ifndef FMCA_H2MATRIX_EIGENH2MATRIXWRAPPER_H_
#define FMCA_H2MATRIX_EIGENH2MATRIXWRAPPER_H_

#include <Eigen/Sparse>

namespace Eigen {

class H2MatrixWrapper;

namespace internal {
// MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
template <>
struct traits<H2MatrixWrapper>
    : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
} // namespace internal

class H2MatrixWrapper : public Eigen::EigenBase<H2MatrixWrapper> {
public:
  typedef FMCA::H2Matrix<FMCA::H2ClusterTree<FMCA::ClusterTreeMesh>> H2Matrix;
  H2MatrixWrapper(const H2Matrix &M) : M_(M){};
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef Scalar RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  Index rows() const { return M_.rows(); }
  Index cols() const { return M_.cols(); }
  const H2Matrix &mat() const { return M_; }

  template <typename Rhs>
  Eigen::Product<H2MatrixWrapper, Rhs, Eigen::AliasFreeProduct>
  operator*(const Eigen::MatrixBase<Rhs> &x) const {
    return Eigen::Product<H2MatrixWrapper, Rhs, Eigen::AliasFreeProduct>(
        *this, x.derived());
  }

private:
  const H2Matrix &M_;
};

} // namespace Eigen

namespace Eigen {
namespace internal {

template <typename Rhs>
struct generic_product_impl<H2MatrixWrapper, Rhs, SparseShape, DenseShape,
                            GemvProduct> // GEMV stands for matrix-vector
    : generic_product_impl_base<H2MatrixWrapper, Rhs,
                                generic_product_impl<H2MatrixWrapper, Rhs>> {
  typedef typename Product<H2MatrixWrapper, Rhs>::Scalar Scalar;

  template <typename Dest>
  static void scaleAndAddTo(Dest &dst, const H2MatrixWrapper &lhs,
                            const Rhs &rhs, const Scalar &alpha) {
    // This method should implement "dst += alpha * lhs * rhs" inplace,
    // however, for iterative solvers, alpha is always equal to 1, so let's not
    // bother about it.
    assert(alpha == Scalar(1) && "scaling is not implemented");
    EIGEN_ONLY_USED_FOR_DEBUG(alpha);

    // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
    // but let's do something fancier (and less efficient):
    dst += lhs.mat() * rhs;
  }
};

} // namespace internal
} // namespace Eigen

#endif
