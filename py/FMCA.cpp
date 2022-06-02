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

#include <Eigen/Dense>
#include <iostream>
//
#include <FMCA/Clustering>

namespace py = pybind11;

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
}
