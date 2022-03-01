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
#include <Eigen/Sparse>
#include <FMCA/Clustering>
#include <FMCA/H2Matrix>
#include <FMCA/MatrixEvaluators>
#include <algorithm>
#include <iostream>
#include <random>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/IO.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>

#include <FMCA/BEM>
#include <FMCA/Moments>

struct harmonicfun {
  template <typename Derived>
  double operator()(const Eigen::MatrixBase<Derived> &x) const {
    return x(0) * x(0) - x(1) * x(1);
  }
};
////////////////////////////////////////////////////////////////////////////////
using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using Moments = FMCA::GalerkinMoments<Interpolator>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTreeMesh>;
using MatrixEvaluator = FMCA::GalerkinMatrixEvaluatorSL<Moments>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const unsigned int level = atoi(argv[1]);
  const std::string fname = "sphere" + std::to_string(level) + ".obj";
  const auto fun = harmonicfun();
  FMCA::Tictoc T;
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::cout << std::string(60, '-') << std::endl;
  std::cout << fname << std::endl;
  // read mesh
  igl::readOBJ("sphere" + std::to_string(level) + ".obj", V, F);
  FMCA::ClusterTreeMesh CT(V, F, 10);
  Moments gal_mom(V, F, 3);
  FMCA::GalerkinRHSEvaluator<Moments> rhs_eval(gal_mom);
  FMCA::SLPotentialEvaluator<Moments> pot_eval(gal_mom);
  MatrixEvaluator mat_eval(gal_mom);
  Eigen::MatrixXd S;
  T.tic();
  mat_eval.compute_dense_block(CT, CT, &S);
  T.toc("matrix assembly: ");
  rhs_eval.compute_rhs(CT, fun);

  Eigen::VectorXd rho = S.lu().solve(rhs_eval.rhs_);
  Eigen::VectorXd pot =
      pot_eval.compute(CT, rho, Eigen::Vector3d({0.1, 0.2, 0.0}));
  std::cout << "error: " <<
    abs(pot(0) - fun(Eigen::Vector3d({0.1, 0.2, 0.0}))) << std::endl;

  Eigen::VectorXd colrs(V.rows());
  for (auto i = 0; i < V.rows(); ++i) colrs(i) = fun(V.row(i));
  Eigen::VectorXd srho(rho.size());
  for (auto i = 0; i < srho.size(); ++i) srho(CT.indices()[i]) = rho(i);
  // FMCA::IO::plotTriMeshColor("bunny.vtk", V.transpose(), F, colrs);
  FMCA::IO::plotTriMeshColor2("result.vtk", V.transpose(), F, srho);
  std::cout << std::string(60, '-') << std::endl;
  return 0;
}
