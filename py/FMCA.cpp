// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
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
#include <FMCA/src/Samplets/omp_samplet_compressor.h>
////////////////////////////////////////////////////////////////////////////////
namespace py = pybind11;

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

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
  FMCA::Index dtilde_;
  SampletTree ST_;
};
////////////////////////////////////////////////////////////////////////////////

/**
 *  \brief wrapper class for an H2 samplet tree
 *
 **/
struct pyH2SampletTree {
  pyH2SampletTree(){};
  pyH2SampletTree(const FMCA::Matrix &P, FMCA::Index dtilde, FMCA::Index mp_deg)
      : dtilde_(dtilde), mp_deg_(mp_deg) {
    const SampletMoments samp_mom(P, dtilde > 0 ? dtilde - 1 : 0);
    const Moments mom(P, mp_deg);
    H2ST_.init(mom, samp_mom, 0, P);
  };
  FMCA::iVector indices() {
    return Eigen::Map<const FMCA::iVector>(H2ST_.indices().data(),
                                           H2ST_.indices().size());
  }
  H2SampletTree H2ST_;
  FMCA::Index dtilde_;
  FMCA::Index mp_deg_;
};
////////////////////////////////////////////////////////////////////////////////

/**
 *  \brief wrapper class for an H2 samplet tree
 *
 **/
struct pyCovarianceKernel {
  pyCovarianceKernel(){};
  pyCovarianceKernel(const std::string &ktype, FMCA::Scalar l) : l_(l) {
    // transform string to upper and check if kernel is implemented
    ktype_ = ktype;
    for (auto &c : ktype_)
      c = (char)toupper(c);
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
  };
  pyPivotedCholesky(const pyCovarianceKernel &ker, const FMCA::Matrix &P,
                    FMCA::Scalar tol = 1e-3)
      : tol_(tol) {
    L_.resize(0, 0);
    B_.resize(0, 0);
    indices_.resize(0);
    compute(ker, P);
  };

  void compute(const pyCovarianceKernel &ker, const FMCA::Matrix &P) {
    // we cap the maximum matrix size at 2GB
    const FMCA::Index max_size = 250000000;
    const FMCA::Index dim = P.cols();
    const FMCA::Index max_cols = max_size / dim > dim ? dim : max_size / dim;
    FMCA::Vector D(dim);
    FMCA::Index pivot = 0;
    FMCA::Scalar tr = 0;
    FMCA::Scalar tol = tol_;
    L_.resize(dim, max_cols);
    indices_.resize(max_cols);
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

  const FMCA::Matrix &matrixB() { return B_; }
  const FMCA::Matrix &matrixL() { return L_; }
  const FMCA::iVector &indices() { return indices_; }
  const FMCA::Scalar &tol() { return tol_; }
  const FMCA::Index &info() { return info_; }

  FMCA::Matrix L_;
  FMCA::Matrix B_;
  FMCA::iVector indices_;
  FMCA::Scalar tol_;
  FMCA::Index info_;
};
////////////////////////////////////////////////////////////////////////////////

// now we just write a matrix evaluator for the covariance kernel
using MatrixEvaluator =
    FMCA::NystromMatrixEvaluator<Moments, pyCovarianceKernel>;

////////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(FMCA, m) {
  m.doc() = "pybind11 FMCA plugin"; // optional module docstring
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
  //////////////////////////////////////////////////////////////////////////////
  // H2sampletTree
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pyH2SampletTree> pyH2SampletTree_(m, "H2SampletTree");
  pyH2SampletTree_.def(py::init<>());
  pyH2SampletTree_.def(
      py::init<const FMCA::Matrix &, FMCA::Index, FMCA::Index>());
  pyH2SampletTree_.def("indices", &pyH2SampletTree::indices);

  m.def(
      "sampletTreeStatistics",
      [](const pyH2SampletTree &tree, const FMCA::Matrix &P) {
        return FMCA::clusterTreeStatistics(tree.H2ST_, P);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Displays metrics of an H2 samplet tree");

  m.def(
      "sampletTransform",
      [](const pyH2SampletTree &tree, const FMCA::Matrix &data) {
        return tree.H2ST_.sampletTransform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Performs samplet transform of data");

  m.def(
      "inverseSampletTransform",
      [](const pyH2SampletTree &tree, const FMCA::Matrix &data) {
        return tree.H2ST_.inverseSampletTransform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Performs inverse samplet transform of data");
  //////////////////////////////////////////////////////////////////////////////
  // CovarianceKernel
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pyCovarianceKernel> pyCovarianceKernel_(m, "CovarianceKernel");
  pyCovarianceKernel_.def(py::init<>());
  pyCovarianceKernel_.def(py::init<const std::string &, FMCA::Scalar>());
  pyCovarianceKernel_.def("kernelType", &pyCovarianceKernel::kernelType);
  pyCovarianceKernel_.def("eval", &pyCovarianceKernel::eval,
                          py::arg().noconvert(), py::arg().noconvert());

  m.def(
      "sampletCompressKernel",
      [](const pyCovarianceKernel &ker, pyH2SampletTree &tree,
         const FMCA::Matrix &P, FMCA::Scalar eta,
         FMCA::Scalar thres) -> Eigen::SparseMatrix<FMCA::Scalar> {
        std::cout << "dtilde: " << tree.dtilde_ << " mpdeg: " << tree.mp_deg_
                  << " eta: " << eta << " threshold: " << thres << std::endl;
        FMCA::ompSampletCompressor<H2SampletTree> comp;
        comp.init(tree.H2ST_, eta, thres);
        const Moments mom(P, tree.mp_deg_);
        const MatrixEvaluator mat_eval(mom, ker);
        Eigen::SparseMatrix<FMCA::Scalar> K(tree.H2ST_.indices().size(),
                                            tree.H2ST_.indices().size());
        comp.compress(tree.H2ST_, mat_eval);
        const auto &trips = comp.triplets();
        K.setFromTriplets(trips.begin(), trips.end());
        return K;
      },
      py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
      py::arg(), py::arg(), "Performs a samplet compression of a kernel");

  //////////////////////////////////////////////////////////////////////////////
  //
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pyPivotedCholesky> pyPivotedCholesky_(m, "PivotedCholesky");
  pyPivotedCholesky_.def(py::init<>());
  pyPivotedCholesky_.def(py::init<const pyCovarianceKernel &,
                                  const FMCA::Matrix &, FMCA::Scalar>());
  pyPivotedCholesky_.def("indices", &pyPivotedCholesky::indices);
  pyPivotedCholesky_.def("matrixL", &pyPivotedCholesky::matrixL);
}
