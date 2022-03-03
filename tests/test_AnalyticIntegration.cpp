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

#include "Quadrature/Quadrature"
#define double float
using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using Moments = FMCA::GalerkinMoments<Interpolator>;

int main(int argc, char *argv[]) {
  const unsigned int level = atoi(argv[1]);
  const std::string fname = "sphere" + std::to_string(level) + ".obj";
  Bembel::GaussSquare<50> Q;
  for (auto i = 0; i < 50; ++i) {
    for (auto j = 0; j < Q[i].xi_.cols(); ++j) {
      Q[i].w_(j) *= (1 - Q[i].xi_(0, j));
      Q[i].xi_(1, j) = (1 - Q[i].xi_(0, j)) * Q[i].xi_(1, j);
    }
    if (i < 4) {
      std::cout << Q[i].w_.transpose() << std::endl;
      std::cout << Q[i].w_.sum() << std::endl;
      std::cout << Q[i].xi_ << std::endl;
      std::cout << Q[i].xi_.colwise().sum() << std::endl;
      std::cout << "---------\n";
    }
  }
  FMCA::Tictoc T;
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  const FMCA::Quad::Quadrature<FMCA::Quad::Radon> Rq;
  const FMCA::Quad::Quadrature<FMCA::Quad::Radon> Mq;
  FMCA::Quad::Quadrature<FMCA::Quad::Trapezoidal> Tq;
  Tq.xi = Q[20].xi_;
  Tq.w = Q[20].w_;
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
  err = 0;
  //////////////////////////////////////////////////////////////////////////////
  for (auto colit : ct.indices()) {
    const FMCA::TriangularPanel &el1 = mom.elements()[colit];
    for (auto rowit : ct.indices()) {
      const FMCA::TriangularPanel &el2 = mom.elements()[rowit];
      if ((el1.mp_ - el2.mp_).norm() > 1.9) {
        analytic_value = 0;
        value = 0;
        for (auto k = 0; k < Tq.xi.cols(); ++k) {
          const Eigen::Vector3d qp2 =
              el2.affmap_.col(0) + el2.affmap_.rightCols(2) * Tq.xi.col(k);
          analytic_value += Tq.w(k) * analyticIntD(el1, qp2);
          for (auto l = 0; l < Tq.xi.cols(); ++l) {
            const Eigen::Vector3d qp1 =
                el1.affmap_.col(0) + el1.affmap_.rightCols(2) * Tq.xi.col(l);
            const double r = std::pow((qp2 - qp1).norm(), 3.);
            const double num = (qp2 - qp1).dot(el1.cs_.col(2));
            value += Tq.w(k) * Tq.w(l) * num / r;
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
      Eigen::Vector3d({0, 0, 0}), Eigen::Vector3d({0.05, 0, 0}),
      Eigen::Vector3d({0.05, 0.05, 0}));
  const FMCA::TriangularPanel el2 = FMCA::TriangularPanel(
      Eigen::Vector3d({2, 0, 0}), Eigen::Vector3d({2.05, 0, 0}),
      Eigen::Vector3d({2.05, 0, 0.05}));
  std::cout << el1.mp_ - el2.mp_ << std::endl;
  dist = (el1.mp_ - el2.mp_).norm();
  std::cout << el1.cs_ << std::endl;
  std::cout << el2.cs_ << std::endl;
  std::cout << el1.volel_ << std::endl;
  std::cout << el2.volel_ << std::endl;

  for (auto k = 0; k < Tq.xi.cols(); ++k) {
    const Eigen::Vector3d qp2 =
        el2.affmap_.col(0) + el2.affmap_.rightCols(2) * Tq.xi.col(k);
    analytic_value += Tq.w(k) * analyticIntD(el1, qp2);
  }
  analytic_value *= 2 * sqrt(el2.volel_) / sqrt(el1.volel_);
  double val3 = 0;
  for (auto k = 0; k < Tq.xi.cols(); ++k) {
    const Eigen::Vector3d qp2 =
        el2.affmap_.col(0) + el2.affmap_.rightCols(2) * Tq.xi.col(k);
    for (auto l = 0; l < Tq.xi.cols(); ++l) {
      const Eigen::Vector3d qp1 =
          el1.affmap_.col(0) + el1.affmap_.rightCols(2) * Tq.xi.col(l);
      const double r = std::pow((qp2 - qp1).norm(), 3.);
      const double num = (qp2 - qp1).dot(el1.cs_.col(2));
      val3 += Tq.w(k) * Tq.w(l) * num / r;
    }
  }
  val3 *= 2 * sqrt(el1.volel_) * sqrt(el2.volel_);
  double val4 = 0;
  for (auto k = 0; k < Rq.xi.cols(); ++k) {
    const Eigen::Vector3d qp2 =
        el2.affmap_.col(0) + el2.affmap_.rightCols(2) * Rq.xi.col(k);
    for (auto l = 0; l < Rq.xi.cols(); ++l) {
      const Eigen::Vector3d qp1 =
          el1.affmap_.col(0) + el1.affmap_.rightCols(2) * Rq.xi.col(l);
      const double r = std::pow((qp2 - qp1).norm(), 3.);
      const double num = (qp2 - qp1).dot(el1.cs_.col(2));
      val4 += Rq.w(k) * Rq.w(l) * num / r;
    }
  }
  val4 *= 2 * sqrt(el1.volel_) * sqrt(el2.volel_);

  value = 0.5 * 0.5 * (el2.mp_ - el1.mp_).dot(el1.cs_.col(2)) /
          std::pow((el2.mp_ - el1.mp_).norm(), 3.) * 2 * sqrt(el1.volel_) *
          sqrt(el2.volel_);
  std::cout << "res ana: " << analytic_value << std::endl;
  std::cout << "res quad: " << val4 << std::endl;
  std::cout << "res quad Gauss: " << val3 << std::endl;
  err = abs(val3 - analytic_value) / abs(val3);
  dist = (el1.mp_ - el2.mp_).norm();
  std::cout << abs(val3-val4) / abs(val3) << std::endl;
  std::cout << "dist: " << dist << std::endl;
  std::cout << "error DL formula same plane: " << err << std::endl;

  return 0;
}
