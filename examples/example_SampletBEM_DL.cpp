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
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
using EigenSparseLU =
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::MetisOrdering<int>>;
#include <FMCA/Clustering>
#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>
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
#include <FMCA/Samplets>

struct harmonicfun {
  template <typename Derived>
  double operator()(const Eigen::MatrixBase<Derived> &x) const {
    return exp(FMCA_PI * x(0)) * sin(FMCA_PI * x(1));
  }
};
////////////////////////////////////////////////////////////////////////////////
using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using Moments = FMCA::CollocationMoments<Interpolator>;
using SampletMoments = FMCA::CollocationSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::CollocationMatrixEvaluatorDL<Moments>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTreeMesh>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const unsigned int level = atoi(argv[1]);
  const std::string fname = "sphere" + std::to_string(level) + ".obj";
  const auto fun = harmonicfun();
  const unsigned int dtilde = 3;
  const double eta = 0.8;
  const unsigned int mp_deg = 4;
  const double threshold = 1e-5;
  FMCA::Tictoc T;
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::cout << std::string(60, '-') << std::endl;
  std::cout << "mesh file: " << fname << std::endl;
  igl::readOBJ("sphere" + std::to_string(level) + ".obj", V, F);
  // igl::readOBJ("bunny.obj", V, F);
  std::cout << "number of elements: " << F.rows() << std::endl;
  const Moments mom(V, F, mp_deg);
  MatrixEvaluator mat_eval(mom);
  SampletMoments samp_mom(V, F, dtilde - 1);
  T.tic();
  H2SampletTree hst(mom, samp_mom, 0, V, F);
  T.toc("tree setup: ");
  std::cout << std::flush;
  FMCA::unsymmetric_compressor_impl<H2SampletTree> Comp;
  T.tic();
  Comp.compress(hst, mat_eval, eta, threshold);
  T.toc("symmetric compressor: ");
  const double tripSize = sizeof(Eigen::Triplet<double>);
  const double nTrips = Comp.pattern_triplets().size();
  std::cout << "nz(S): " << std::ceil(nTrips / V.rows()) << std::endl;
  std::cout << "memory: " << nTrips * tripSize / 1e9 << "GB\n" << std::flush;
  std::cout << std::flush;
  FMCA::CollocationRHSEvaluator<Moments> rhs_eval(mom);
  FMCA::SLCollocationPotentialEvaluator<Moments> pot_eval(mom);
  rhs_eval.compute_rhs(hst, fun);
  Eigen::VectorXd srhs = hst.sampletTransform(rhs_eval.rhs_);

  Eigen::SparseMatrix<double> S(F.rows(), F.rows());
  const auto &trips = Comp.pattern_triplets();
  S.setFromTriplets(trips.begin(), trips.end());
  S = 0.5 * Eigen::MatrixXd::Identity(F.rows(), F.rows()) - S;
  EigenSparseLU solver;
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
  return 0;
}
