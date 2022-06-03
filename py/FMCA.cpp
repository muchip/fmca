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
#include <iostream>
//
#include <FMCA/Clustering>
#include <FMCA/Samplets>

using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;

namespace py = pybind11;

struct pySampletTree {
  pySampletTree(){};
  pySampletTree(const FMCA::Matrix &P, FMCA::Index dtilde) {
    const SampletMoments samp_mom(P, dtilde > 0 ? dtilde - 1 : 0);
    ST_.init(samp_mom, 0, P);
  };
  FMCA::iVector indices() {
    return Eigen::Map<const FMCA::iVector>(ST_.indices().data(),
                                           ST_.indices().size());
  }
  SampletTree ST_;
};

void metrics(Eigen::Ref<FMCA::Matrix> mat) {
  std::cout << mat << std::endl;
  return;
}

PYBIND11_MODULE(FMCA, m) {
  m.doc() = "pybind11 FMCA plugin";  // optional module docstring
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
}
