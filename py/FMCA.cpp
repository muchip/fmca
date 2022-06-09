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
//
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
//
#include <Eigen/Dense>
#include <functional>
#include <iostream>
//
#include <FMCA/Clustering>
#include <FMCA/Samplets>
#include <FMCA/src/Samplets/omp_samplet_compressor.h>

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

namespace py = pybind11;

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
    for (auto i = 0; i < PC.cols(); ++i)
      for (auto j = 0; j < PR.cols(); ++j)
        retval(i, j) = operator()(PC.col(i), PR.col(j));
    return retval;
  }

  std::string kernelType() const { return ktype_; }

  std::function<FMCA::Scalar(FMCA::Scalar)> kernel_;
  std::string ktype_;
  FMCA::Scalar l_;
};

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
      "Displays metrics of a samplet tree");

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
        std::cout << tree.dtilde_ << " " << tree.mp_deg_ << " "
                  << " " << eta << " " << thres << std::endl;
        FMCA::ompSampletCompressor<H2SampletTree> comp;
        comp.init(tree.H2ST_, eta, thres);
        const Moments mom(P, tree.mp_deg_);
        const MatrixEvaluator mat_eval(mom, ker);
        Eigen::SparseMatrix<FMCA::Scalar> K(tree.H2ST_.indices().size(),
                                            tree.H2ST_.indices().size());
        comp.compress(tree.H2ST_, mat_eval);
        const auto &trips = comp.pattern_triplets();
        K.setFromTriplets(trips.begin(), trips.end());
        return K;
      },
      py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
      py::arg(), py::arg(), "Performs a samplet compression of a kernel");
}
