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
#include <iostream>
#include <string>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/Tictoc.h>
#include <igl/readOBJ.h>

#include <FMCA/Clustering>
#include <FMCA/MatrixEvaluators>

using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using Moments = FMCA::GalerkinMoments<Interpolator>;

int main(int argc, char *argv[]) {
  const unsigned int level = atoi(argv[1]);
  const std::string fname = "sphere" + std::to_string(level) + ".obj";
  FMCA::Tictoc T;
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  const FMCA::Quad::Quadrature<FMCA::Quad::Radon> Rq;
  const FMCA::Quad::Quadrature<FMCA::Quad::Radon> Mq;
  std::cout << std::string(60, '-') << std::endl;
  std::cout << "mesh file: " << fname << std::endl;
  igl::readOBJ("sphere" + std::to_string(level) + ".obj", V, F);
  std::cout << "number of elements: " << F.rows() << std::endl;
  const Moments mom(V, F, 0);
  FMCA::ClusterTreeMesh ct(V, F, 10);
  double err = 0;
  double dist = 0;
  double analytic_value = 0;
  double value = 0;
  for (auto colit : ct.indices()) {
    const FMCA::TriangularPanel &el1 = mom.elements()[colit];
    for (auto rowit : ct.indices()) {
      const FMCA::TriangularPanel &el2 = mom.elements()[rowit];
      if ((el1.mp_ - el2.mp_).norm() > 1.9) {
        analytic_value = 0;
        value = 0;
        for (auto k = 0; k < Rq.xi.cols(); ++k) {
          const Eigen::Vector3d qp2 =
              el2.affmap_.col(0) + el2.affmap_.rightCols(2) * Rq.xi.col(k);
          analytic_value += Rq.w(k) * analyticIntS(el1, qp2);
          for (auto l = 0; l < Rq.xi.cols(); ++l) {
            const Eigen::Vector3d qp1 =
                el1.affmap_.col(0) + el1.affmap_.rightCols(2) * Rq.xi.col(l);
            const double r = (qp2 - qp1).norm();
            value += Rq.w(k) * Rq.w(l) / r;
          }
        }
        analytic_value *= 2 * sqrt(el2.volel_) / sqrt(el1.volel_);
        value *= 2 * sqrt(el1.volel_) * sqrt(el2.volel_);
        if (abs(value - analytic_value) / abs(analytic_value) > err) {
          err = abs(value - analytic_value) / abs(analytic_value);
          dist = (el1.mp_ - el2.mp_).norm();
        }
      }
    }
  }
  std::cout << dist << std::endl;
  std::cout << "error SL formula: " << err << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  for (auto colit : ct.indices()) {
    const FMCA::TriangularPanel &el1 = mom.elements()[colit];
    for (auto rowit : ct.indices()) {
      const FMCA::TriangularPanel &el2 = mom.elements()[rowit];
      if ((el1.mp_ - el2.mp_).norm() > 1.9) {
        analytic_value = 0;
        value = 0;
        for (auto k = 0; k < Rq.xi.cols(); ++k) {
          const Eigen::Vector3d qp2 =
              el2.affmap_.col(0) + el2.affmap_.rightCols(2) * Rq.xi.col(k);
          analytic_value += Rq.w(k) * analyticIntD(el1, qp2);
          for (auto l = 0; l < Rq.xi.cols(); ++l) {
            const Eigen::Vector3d qp1 =
                el1.affmap_.col(0) + el1.affmap_.rightCols(2) * Rq.xi.col(l);
            const double r = std::pow((qp2 - qp1).norm(), 3.);
            const double num = (qp2 - qp1).dot(el1.cs_.col(2));
            value += Rq.w(k) * Rq.w(l) * num / r;
          }
        }
        analytic_value *= 2 * sqrt(el2.volel_) / sqrt(el1.volel_);
        value *= 2 * sqrt(el1.volel_) * sqrt(el2.volel_);
        if (abs(value - analytic_value) / abs(analytic_value) > err) {
          err = abs(value - analytic_value) / abs(analytic_value);
          dist = (el1.mp_ - el2.mp_).norm();
        }
      }
    }
  }
  std::cout << dist << std::endl;
  std::cout << "error DL formula: " << err << std::endl;
  analytic_value = 0;
  value = 0;
  err = 0;
  const FMCA::TriangularPanel el1 = FMCA::TriangularPanel(
      Eigen::Vector3d({0, 0, 0}), Eigen::Vector3d({1, 0, 0}),
      Eigen::Vector3d({1, 1, 0}));
  const FMCA::TriangularPanel el2 = FMCA::TriangularPanel(
      Eigen::Vector3d({4, 0, 0}), Eigen::Vector3d({5, 0, 0}),
      Eigen::Vector3d({5, 0, 1}));
  std::cout << el1.mp_ - el2.mp_ << std::endl;
  dist = (el1.mp_ - el2.mp_).norm();
  std::cout << el1.cs_ << std::endl;
  std::cout << el2.cs_ << std::endl;
  std::cout << el1.volel_ << std::endl;
  std::cout << el2.volel_ << std::endl;

  if ((el1.mp_ - el2.mp_).norm() > 1.9) {
    analytic_value = 0;
    value = 0;
    for (auto k = 0; k < Rq.xi.cols(); ++k) {
      const Eigen::Vector3d qp2 =
          el2.affmap_.col(0) + el2.affmap_.rightCols(2) * Rq.xi.col(k);
      analytic_value += Rq.w(k) * analyticIntD(el1, el2.mp_);
    }
    analytic_value *= el2.volel_;

    for (auto l = 0; l < Rq.xi.cols(); ++l) {
      const Eigen::Vector3d qp1 =
          el1.affmap_.col(0) + el1.affmap_.rightCols(2) * Rq.xi.col(l);
      const double r = std::pow((qp2 - qp1).norm(), 3.);
      const double num = (qp2 - qp1).dot(el1.cs_.col(2));
      value += Rq.w(k) * Rq.w(l) * num / r;
    }
    std::cout << "res ana: " << analytic_value << std::endl;
    std::cout << "res quad: " << value << std::endl;
    value *= el1.volel_ * el2.volel_;
    if (abs(value - analytic_value) / abs(analytic_value) > err) {
      err = abs(value - analytic_value) / abs(analytic_value);
      dist = (el1.mp_ - el2.mp_).norm();
    }
  }
  std::cout << "dist: " << dist << std::endl;
  std::cout << "error DL formula same plane: " << err << std::endl;

  return 0;
}
