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
#include <FMCA/Moments>
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/IO.h>
#include <FMCA/src/util/Tictoc.h>
////////////////////////////////////////////////////////////////////////////////
using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using Moments = FMCA::GalerkinMoments<Interpolator>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTreeMesh>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  // read mesh
  igl::readOBJ("bunny.obj", V, F);
  FMCA::ClusterTreeMesh CT(V, F, 10);
  Moments gal_mom(V, F, 3);
  for (auto level = 0; level < 16; ++level) {
    std::vector<Eigen::MatrixXd> bbvec;
    for (auto &node : CT) {
      if (node.level() == level)
        bbvec.push_back(node.derived().bb());
    }
    FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
  }

  std::vector<int> map(CT.indices().size());
  std::iota(map.begin(), map.end(), 0);
  std::shuffle(map.begin(), map.end(), std::default_random_engine(0));

  Eigen::VectorXd colrs(V.rows());
  for (auto &node : CT) {
    if (!node.nSons()) {
      for (auto it = node.indices().begin(); it != node.indices().end(); ++it)
        for (auto j = 0; j < F.cols(); ++j)
          colrs(F(*it, j)) = map[node.block_id()];
    }
  }

  FMCA::IO::plotTriMeshColor("bunny.vtk", V.transpose(), F, colrs);
  return 0;
}
