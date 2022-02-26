#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <FMCA/Clustering>
#include <algorithm>
#include <igl/readOBJ.h>
#include <iostream>
#include <random>

#include "../FMCA/src/BEM/NumericalQuadrature.h"

#include "../FMCA/src/util/IO.h"

int main(int argc, char *argv[]) {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  FMCA::Quad::Quadrature<FMCA::Quad::Trapezoidal> Rq;
  std::cout << Rq.w.transpose() << std::endl;
  std::cout << Rq.w.sum() << std::endl;
  std::cout << Rq.xi << std::endl;
  // read mesh
  igl::readOBJ("bunny.obj", V, F);
  FMCA::ClusterTreeMesh CT(V, F, 100);

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
