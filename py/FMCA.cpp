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
////////////////////////////////////////////////////////////////////////////////
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <functional>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/Clustering>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
namespace py = pybind11;

using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;

/**
 *  \brief wrapper class for a samplet tree
 *
 **/
struct pySampletTree {
  pySampletTree(){};
  pySampletTree(const FMCA::Matrix &P, FMCA::Index dtilde) : dtilde_(dtilde) {
    const SampletMoments samp_mom(P, dtilde > 0 ? dtilde - 1 : 0);
    ST_.init(samp_mom, 0, P);
  };
  FMCA::iVector indices() {
    return Eigen::Map<const FMCA::iVector>(ST_.indices().data(),
                                           ST_.indices().size());
  }
  FMCA::iVector levels() {
    std::vector<FMCA::Index> lvl = FMCA::internal::sampletLevelMapper(ST_);
    return Eigen::Map<const FMCA::iVector>(lvl.data(), lvl.size());
  }
  FMCA::Index dtilde_;
  SampletTree ST_;
};
////////////////////////////////////////////////////////////////////////////////

/**
 *  \brief wrapper class for covariance kernel
 *
 **/
struct pyCovarianceKernel {
  pyCovarianceKernel(){};
  pyCovarianceKernel(const std::string &ktype, FMCA::Scalar l) : l_(l) {
    // transform string to upper and check if kernel is implemented
    ktype_ = ktype;
    for (auto &c : ktype_) c = (char)toupper(c);
    if (ktype_ == "GAUSSIAN")
      kernel_ = [this](FMCA::Scalar r) { return exp(-r * r / l_); };
    else if (ktype_ == "EXPONENTIAL")
      kernel_ = [this](FMCA::Scalar r) { return exp(-r / l_); };
    else
      assert(false && "desired kernel not implemented");
  }

  template <typename derived, typename otherDerived>
  FMCA::Scalar operator()(const Eigen::MatrixBase<derived> &x,
                          const Eigen::MatrixBase<otherDerived> &y) const {
    return kernel_((x - y).norm());
  }
  FMCA::Matrix eval(const FMCA::Matrix &PR, const FMCA::Matrix &PC) const {
    FMCA::Matrix retval(PR.cols(), PC.cols());
    for (auto j = 0; j < PC.cols(); ++j)
      for (auto i = 0; i < PR.cols(); ++i)
        retval(i, j) = operator()(PR.col(i), PC.col(j));
    return retval;
  }

  std::string kernelType() const { return ktype_; }

  std::function<FMCA::Scalar(FMCA::Scalar)> kernel_;
  std::string ktype_;
  FMCA::Scalar l_;
};
////////////////////////////////////////////////////////////////////////////////

/**
 *  \brief class providing Cholesky kernel approximations
 *
 **/
struct pyPivotedCholesky {
  pyPivotedCholesky() {
    L_.resize(0, 0);
    B_.resize(0, 0);
    indices_.resize(0);
    tol_ = 0;
    mineig_ = 0;
  };
  pyPivotedCholesky(const pyCovarianceKernel &ker, const FMCA::Matrix &P,
                    FMCA::Scalar tol = 1e-3)
      : tol_(tol) {
    L_.resize(0, 0);
    B_.resize(0, 0);
    indices_.resize(0);
    mineig_ = 0;
    compute(ker, P, tol);
  };

  void compute(const pyCovarianceKernel &ker, const FMCA::Matrix &P,
               FMCA::Scalar tol = 1e-3) {
    const FMCA::Index dim = P.cols();
    const FMCA::Index max_cols = max_size_ / dim > dim ? dim : max_size_ / dim;
    FMCA::Vector D(dim);
    FMCA::Index pivot = 0;
    FMCA::Scalar tr = 0;
    L_.resize(dim, max_cols);
    indices_.resize(max_cols);
    tol_ = tol;
    // compute the diagonal and the trace
    for (auto i = 0; i < dim; ++i) {
      const FMCA::Matrix wtf = ker.eval(P.col(i), P.col(i));
      D(i) = wtf(0, 0);
      if (D(i) < 0) {
        info_ = 1;
        return;
      }
    }
    tr = D.sum();
    // we guarantee the error tr(A-LL^T)/tr(A) < tol
    tol *= tr;
    // perform pivoted Cholesky decomposition
    std::cout << "N: " << dim << " max number of cols: " << max_cols
              << " rel tol: " << tol << " initial trace: " << tr << std::endl;
    FMCA::Index step = 0;
    while ((step < max_cols) && (tol < tr)) {
      D.maxCoeff(&pivot);
      indices_(step) = pivot;
      // get new column from C
      L_.col(step) = ker.eval(P, P.col(pivot));
      // update column with the current matrix Lmatrix_
      L_.col(step) -= L_.leftCols(step) * L_.row(pivot).head(step).transpose();
      if (L_(pivot, step) <= 0) {
        info_ = 2;
        std::cout << "breaking with non positive pivot\n";
        break;
      }
      L_.col(step) /= sqrt(L_(pivot, step));
      // update the diagonal and the trace
      D.array() -= L_.col(step).array().square();
      // compute the trace of the Schur complement
      tr = D.sum();
      ++step;
    }
    std::cout << "steps: " << step << " trace error: " << tr << std::endl;
    if (tr < 0)
      info_ = 2;
    else
      info_ = 0;
    // crop L, indices to their actual size
    L_.conservativeResize(dim, step);
    indices_.conservativeResize(step);
    return;
  }

  void computeFullPiv(const pyCovarianceKernel &ker, const FMCA::Matrix &P,
                      FMCA::Scalar tol = 1e-3) {
    const FMCA::Index dim = P.cols();
    const FMCA::Index max_cols = max_size_ / dim > dim ? dim : max_size_ / dim;
    tol_ = tol;
    if (max_cols < dim) {
      info_ = 3;
      return;
    }
    Eigen::SelfAdjointEigenSolver<FMCA::Matrix> es;
    {
      FMCA::Matrix K = ker.eval(P, P);
      es.compute(K);
      info_ = es.info();
      if (es.info() != Eigen::Success) return;
    }
    FMCA::Vector ev = es.eigenvalues().reverse();
    std::cout << "lambda min: " << ev.minCoeff() << " "
              << "lambda max: " << ev.maxCoeff();
    mineig_ = ev.minCoeff();
    FMCA::Scalar tr = ev.sum();
    FMCA::Scalar cur_tr = 0;
    FMCA::Index step = 0;
    while (tr - cur_tr > tol * tr) {
      cur_tr += ev(step);
      ++step;
    }
    std::cout << " step: " << step << std::endl;
    L_.resize(dim, step);
    for (auto i = 1; i <= step; ++i)
      L_.col(i - 1) = es.eigenvectors().col(dim - i);
    L_ = L_ * ev.head(step).cwiseSqrt().asDiagonal();
    return;
  }

  const FMCA::Matrix &matrixB() { return B_; }
  const FMCA::Matrix &matrixL() { return L_; }
  const FMCA::iVector &indices() { return indices_; }
  const FMCA::Scalar &tol() { return tol_; }
  const FMCA::Index &info() { return info_; }
  const FMCA::Scalar& mineig() { return mineig_; }

  FMCA::Matrix L_;
  FMCA::Matrix B_;
  FMCA::iVector indices_;
  FMCA::Scalar tol_;
  FMCA::Index info_;
  FMCA::Scalar mineig_;
  // we cap the maximum matrix size at 2GB
  const FMCA::Index max_size_ = 250000000;
};
////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(FMCA, m) {
  m.doc() = "pybind11 FMCA plugin";  // optional module docstring
  //////////////////////////////////////////////////////////////////////////////
  // ClusterTree
  //////////////////////////////////////////////////////////////////////////////
  py::class_<FMCA::ClusterTree> ClusterTree_(m, "ClusterTree");
  ClusterTree_.def(py::init<>());
  ClusterTree_.def(py::init<const FMCA::Matrix &, FMCA::Index>());
  m.def(
      "clusterTreeStatistics",
      [](const FMCA::ClusterTree &tree, const FMCA::Matrix &P) {
        return FMCA::clusterTreeStatistics(tree, P);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Displays metrics of a cluster tree");
  //////////////////////////////////////////////////////////////////////////////
  // SampletTree
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pySampletTree> pySampletTree_(m, "SampletTree");
  pySampletTree_.def(py::init<>());
  pySampletTree_.def(py::init<const FMCA::Matrix &, FMCA::Index>());
  pySampletTree_.def("indices", &pySampletTree::indices);
  pySampletTree_.def("levels", &pySampletTree::levels);

  m.def(
      "sampletTreeStatistics",
      [](const pySampletTree &tree, const FMCA::Matrix &P) {
        return FMCA::clusterTreeStatistics(tree.ST_, P);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Displays metrics of a samplet tree");

  m.def(
      "sampletTransform",
      [](const pySampletTree &tree, const FMCA::Matrix &data) {
        return tree.ST_.sampletTransform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Performs samplet transform of data");

  m.def(
      "inverseSampletTransform",
      [](const pySampletTree &tree, const FMCA::Matrix &data) {
        return tree.ST_.inverseSampletTransform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Performs inverse samplet transform of data");

  m.def(
      "sampletTransformMinLevel",
      [](const pySampletTree &tree, const FMCA::Matrix &data,
         const FMCA::Index min_level) {
        FMCA::SampletTransformer<SampletTree> s_trafo(tree.ST_, min_level);
        return s_trafo.transform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
      "Performs samplet transform of data");
  //////////////////////////////////////////////////////////////////////////////
  // CovarianceKernel
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pyCovarianceKernel> pyCovarianceKernel_(m, "CovarianceKernel");
  pyCovarianceKernel_.def(py::init<>());
  pyCovarianceKernel_.def(py::init<const std::string &, FMCA::Scalar>());
  pyCovarianceKernel_.def("kernelType", &pyCovarianceKernel::kernelType);
  pyCovarianceKernel_.def("eval", &pyCovarianceKernel::eval,
                          py::arg().noconvert(), py::arg().noconvert());
  //////////////////////////////////////////////////////////////////////////////
  // pivoted Cholesky decomposition
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pyPivotedCholesky> pyPivotedCholesky_(m, "PivotedCholesky");
  pyPivotedCholesky_.def(py::init<>());
  pyPivotedCholesky_.def(py::init<const pyCovarianceKernel &,
                                  const FMCA::Matrix &, FMCA::Scalar>());
  pyPivotedCholesky_.def("compute", &pyPivotedCholesky::compute,
                         py::arg().noconvert(), py::arg().noconvert(),
                         py::arg(),
                         "Computes the pivoted Cholesky decomposition");
  pyPivotedCholesky_.def("computeFullPiv", &pyPivotedCholesky::computeFullPiv,
                         py::arg().noconvert(), py::arg().noconvert(),
                         py::arg(),
                         "Computes the truncated spectral decomposition");
  pyPivotedCholesky_.def("indices", &pyPivotedCholesky::indices);
  pyPivotedCholesky_.def("matrixL", &pyPivotedCholesky::matrixL);
  pyPivotedCholesky_.def("mineig", &pyPivotedCholesky::mineig);

}
