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
#include <Eigen/IterativeLinearSolvers>
#include <FMCA/Clustering>
#include <FMCA/H2Matrix>
#include <FMCA/MatrixEvaluators>
#include <algorithm>
#include <iostream>
#include <random>
#include <unsupported/Eigen/IterativeSolvers>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/IO.h>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>

#include <FMCA/BEM>
#include <FMCA/H2Matrix>
#include <FMCA/src/H2Matrix/EigenH2MatrixWrapper.h>

#include <FMCA/Moments>

struct harmonicfun {
  template <typename Derived>
  double operator()(const Eigen::MatrixBase<Derived> &x) const {
    return exp(FMCA_PI * x(0)) * sin(FMCA_PI * x(1));
  }
};
////////////////////////////////////////////////////////////////////////////////
using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using Moments = FMCA::CollocationMoments<Interpolator>;
using MatrixEvaluator = FMCA::CollocationMatrixEvaluatorSL<Moments>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTreeMesh>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const unsigned int level = atoi(argv[1]);
  const std::string fname = "sphere" + std::to_string(level) + ".obj";
  const auto fun = harmonicfun();
  const double eta = 0.8;
  const unsigned int mp_deg = 4;
  FMCA::Tictoc T;
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::cout << std::string(60, '-') << std::endl;
  std::cout << "mesh file: " << fname << std::endl;
  igl::readOBJ(fname, V, F);
  // igl::readOBJ("bunny.obj", V, F);
  std::cout << "number of elements: " << F.rows() << std::endl;
  T.tic();
  const Moments mom(V, F, mp_deg);
  const H2ClusterTree ct(mom, 0, V, F);
  const MatrixEvaluator mat_eval(mom);
  T.toc("tree setup: ");
  T.tic();
  H2Matrix hmat(ct, mat_eval, eta);
  T.toc("H2matrix setup: ");
  Eigen::H2MatrixWrapper EigenH2(hmat);
  FMCA::CollocationRHSEvaluator<Moments> rhs_eval(mom);
  FMCA::SLCollocationPotentialEvaluator<Moments> pot_eval(mom);
  T.tic();
  Eigen::BiCGSTAB<Eigen::H2MatrixWrapper, Eigen::IdentityPreconditioner> bicg;
  bicg.compute(EigenH2);
  rhs_eval.compute_rhs(ct, fun);
  Eigen::VectorXd rho = bicg.solve(rhs_eval.rhs_);
  T.toc("solver time: ");
  std::cout << "bicg: #iterations: " << bicg.iterations()
            << " error: " << bicg.error() << std::endl;
  Eigen::MatrixXd pot_pts = Eigen::MatrixXd::Random(V.cols(), 100) * 0.25;
  Eigen::VectorXd pot = pot_eval.compute(ct, rho, pot_pts);
  Eigen::VectorXd exact_vals = pot;
  for (auto i = 0; i < exact_vals.size(); ++i)
    exact_vals(i) = fun(pot_pts.col(i));
  std::cout << "potential error: " << (pot - exact_vals).cwiseAbs().maxCoeff()
            << std::endl;

#if 0
  const double tripSize = sizeof(Eigen::Triplet<double>);
  const double nTrips = symComp.pattern_triplets().size();
  std::cout << "nz(S): " << std::ceil(nTrips / V.rows()) << std::endl;
  std::cout << "memory: " << nTrips * tripSize / 1e9 << "GB\n" << std::flush;
  std::cout << std::flush;
  FMCA::CollocationRHSEvaluator<Moments> rhs_eval(mom);
  FMCA::SLCollocationPotentialEvaluator<Moments> pot_eval(mom);
  rhs_eval.compute_rhs(hst, fun);
  Eigen::VectorXd srhs = hst.sampletTransform(rhs_eval.rhs_);

  Eigen::SparseMatrix<double> S(F.rows(), F.rows());
  const auto &trips = symComp.pattern_triplets();
  S.setFromTriplets(trips.begin(), trips.end());
  FMCA::IO::print2m("stiff.m", "S", S, "w");
  EigenCholesky solver;
  T.tic();
  solver.compute(S);
  T.toc("time factorization: ");
  std::cout << "sinfo: " << solver.info() << std::endl;
  Eigen::VectorXd srho = solver.solve(srhs);
  Eigen::VectorXd rho = hst.inverseSampletTransform(srho);

  std::cout << "norm rho: " << rho.norm() << " " << rho.sum() << std::endl;
  Eigen::MatrixXd pot_pts = Eigen::MatrixXd::Random(V.cols(), 100) * 0.25;
  Eigen::VectorXd pot = pot_eval.compute(hst, rho, pot_pts);
  Eigen::VectorXd exact_vals = pot;
  for (auto i = 0; i < exact_vals.size(); ++i)
    exact_vals(i) = fun(pot_pts.col(i));
  std::cout << "error: " << (pot - exact_vals).cwiseAbs().maxCoeff()
            << std::endl;

  Eigen::VectorXd colrs(V.rows());
  for (auto i = 0; i < V.rows(); ++i)
    colrs(i) = fun(V.row(i));
  Eigen::VectorXd srho2(rho.size());
  for (auto i = 0; i < srho2.size(); ++i)
    srho2(hst.indices()[i]) =
        rho(i) / sqrt(0.5 * mom.elements()[hst.indices()[i]].volel_);
  FMCA::IO::plotTriMeshColor("rhs.vtk", V.transpose(), F, colrs);
  FMCA::IO::plotTriMeshColor2("result.vtk", V.transpose(), F, srho2);
  std::cout << std::string(60, '-') << std::endl;
#endif
  return 0;
}
