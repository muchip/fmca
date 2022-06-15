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
#define FMCA_CLUSTERSET_
#include <iostream>
#include <string>
////////////////////////////////////////////////////////////////////////////////
#include <igl/readOBJ.h>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/IO.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>

#include <FMCA/BEM>
#include <FMCA/Samplets>

struct harmonicFun {
  template <typename Derived>
  double operator()(const Eigen::MatrixBase<Derived> &x) const {
    return exp(FMCA_PI * x(0)) * sin(FMCA_PI * x(1));
  }
};

struct oneFun {
  template <typename Derived>
  double operator()(const Eigen::MatrixBase<Derived> &x) const {
    return 1;
  }
};

struct gradHarmonicFun {
  template <typename Derived>
  Eigen::Vector3d operator()(const Eigen::MatrixBase<Derived> &x) const {
    Eigen::Vector3d retval;
    retval << FMCA_PI * exp(FMCA_PI * x(0)) * sin(FMCA_PI * x(1)),
        FMCA_PI * exp(FMCA_PI * x(0)) * cos(FMCA_PI * x(1)), 0;
    return retval;
  }
};

////////////////////////////////////////////////////////////////////////////////
using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using Moments = FMCA::CollocationMoments<Interpolator>;
using MatrixEvaluatorK = FMCA::CollocationMatrixEvaluatorDL<Moments>;
using MatrixEvaluatorS = FMCA::CollocationMatrixEvaluatorSL<Moments>;
using ClusterTree = FMCA::ClusterTreeMesh;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const unsigned int level = atoi(argv[1]);
  const std::string fname = "sphere" + std::to_string(level) + ".obj";
  const auto fun = harmonicFun();
  const auto one = oneFun();
  const auto gradFun = gradHarmonicFun();
  FMCA::Tictoc T;
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::cout << std::string(60, '-') << std::endl;
  std::cout << "mesh file: " << fname << std::endl;
  igl::readOBJ(fname, V, F);
  // V.col(2).array() += 0.05 * (2 * FMCA_PI * V.col(2)).array().cos();
  // V.col(1).array() *= 1.4;

  std::cout << "number of elements: " << F.rows() << std::endl;
  const Moments mom(V, F, 1);
  ClusterTree ct(V, F, 10);
  MatrixEvaluatorS mat_evalS(mom);
  MatrixEvaluatorK mat_evalK(mom);
  T.tic();
  Eigen::MatrixXd K;
  Eigen::MatrixXd S;
  mat_evalK.compute_dense_block(ct, ct, &K);
  mat_evalS.compute_dense_block(ct, ct, &S);
  S = 0.5 * (S + S.transpose());
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(F.rows(), F.rows());
  T.toc("matrix setup: ");
  //////////////////////////////////////////////////////////////////////////////
  FMCA::CollocationRHSEvaluator<Moments> rhs_eval(mom);
  rhs_eval.compute_rhs(ct, fun);
  Eigen::VectorXd rhs = rhs_eval.rhs_;
  rhs_eval.compute_rhs(ct, one);
  Eigen::VectorXd one_rhs = rhs_eval.rhs_;
  std::cout << ((K + 0.5 * I) * one_rhs).norm() << std::endl;
  FMCA::DLCollocationPotentialEvaluator<Moments> pot_eval(mom);
  Eigen::VectorXd rho = (K - 0.5 * I).householderQr().solve(rhs);


  Eigen::VectorXd rho2 = S.householderQr().solve((K + 0.5 * I) * rhs_eval.rhs_);
  //////////////////////////////////////////////////////////////////////////////
  std::cout << "norm rho: " << rho.norm() << " " << rho.sum() << std::endl;
  Eigen::MatrixXd pot_pts = Eigen::MatrixXd::Random(V.cols(), 100) * 0.25;
  Eigen::VectorXd pot = pot_eval.compute(ct, rho, pot_pts);
  Eigen::VectorXd exact_vals = pot;
  for (auto i = 0; i < exact_vals.size(); ++i)
    exact_vals(i) = fun(pot_pts.col(i));
  std::cout << "error: " << (pot - exact_vals).cwiseAbs().maxCoeff()
            << std::endl;

  Eigen::VectorXd colrs(V.rows());
  Eigen::VectorXd srho(F.rows());
  Eigen::VectorXd refN(F.rows());
  for (auto i = 0; i < V.rows(); ++i)
    colrs(i) = fun(V.row(i));
  double err = 0;
  double ref = 0;
  for (auto i = 0; i < srho.size(); ++i) {
    srho(ct.indices()[i]) =
        rho2(i) / sqrt(0.5 * mom.elements()[ct.indices()[i]].volel_);
    refN(i) = gradFun(mom.elements()[ct.indices()[i]].mp_)
                  .dot(mom.elements()[ct.indices()[i]].cs_.col(2));
    err += 0.5 * (srho(ct.indices()[i]) - refN(i)) *
           (srho(ct.indices()[i]) - refN(i)) *
           mom.elements()[ct.indices()[i]].volel_;
  }
  std::cout << "L2 error Neumann data: " << sqrt(err) << std::endl;
  FMCA::IO::plotTriMeshColor("rhs.vtk", V.transpose(), F, colrs);
  FMCA::IO::plotTriMeshColor2("result.vtk", V.transpose(), F, srho);
  std::cout << std::string(60, '-') << std::endl;
  return 0;
}
