#include <igl/adjacency_list.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <metis.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <random>

#include "../FMCA/src/util/IO.h"

using namespace std;

int main(int argc, char *argv[]) {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  // Load a mesh in OBJ or OFF format
  igl::readOBJ("bunny.obj", V, F);
  std::vector<std::vector<double> > A;
  igl::adjacency_list(F, A);
  // STEP 1: Setup metis data-structures
  std::vector<idx_t> xadj;
  std::vector<idx_t> adjncy;
  int num = 0;
  xadj.push_back(num);
  for (auto item : A) {
    for (auto idx : item) {
      adjncy.push_back(idx);
      num++;
    }
    xadj.push_back(num);
  }
  idx_t nvtxs = xadj.size() - 1;
  idx_t nEdges = adjncy.size() / 2;
  // Number of balancing constraints, which must be at least 1.
  idx_t ncon = 1;
  //  The number of parts requested for the partition.
  idx_t nParts = 1000;
  //  On return, the edge cut volume of the partitioning solution.
  idx_t objval;
  //  On return, the partition vector for the graph.
  idx_t part[nvtxs];
  // Step 2: Partition
  int ret = METIS_PartGraphKway(&nvtxs, &ncon, &xadj[0], &adjncy[0], NULL, NULL,
                                NULL, &nParts, NULL, NULL, NULL, &objval, part);
  // Step 3: Visualization
  Eigen::MatrixXd ids;
  ids.resize(nvtxs, 1);
  for (unsigned part_i = 0; part_i < nvtxs; part_i++) {
    ids(part_i, 0) = part[part_i];
  }
  // Random permutation
  std::vector<int> map;
  for (int i = 0; i < nParts; i++) {
    map.push_back(i);
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(std::begin(map), std::end(map), g);

  for (unsigned part_i = 0; part_i < nvtxs; part_i++) {
    int temp = (int)ids(part_i, 0);
    ids(part_i, 0) = (double)map[temp];
  }
  FMCA::IO::plotPointsColor("bunny.vtk", V.transpose(), ids);
  return 0;
}
