#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <functional>
#include <iomanip>

#include "FMCA/BlockClusterTree"
#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
#include "print2file.hpp"
#include "util/tictoc.hpp"

#include "matrix.h"
#include "mex.h"

//#define NPTS 16384
//#define NPTS 1638400
//#define NPTS 81920
//#define NPTS 8192
#define NPTS 512
#define DIM 3

struct emptyFun {};
using ClusterT = FMCA::ClusterTree<double, DIM, 2>;

/** nlhs Number of expected output mxArrays
 *   plhs Array of pointers to the expected output mxArrays
 *   nrhs Number of input mxArrays
 *   prhs Array of pointers to the input mxArrays.
 *        Do not modify any prhs values in your MEX file.
 *        Changing the data in these read-only mxArrays can
 *        produce undesired side effects.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 0) {
    mexErrMsgIdAndTxt("MATLAB:MXmain:nargin", "MXmain zero. Check help.");
  }
  srand(0);
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  P.row(2).setZero();
  Eigen::VectorXd nrms = P.colwise().norm();
  for (auto i = 0; i < P.cols(); ++i)
    P.col(i) *= 1 / nrms(i);
  tictoc T;
  T.tic();
  ClusterT CT(P);
  FMCA::SampletTree<ClusterT> ST(P, CT, 1);
  T.toc("set up ct: ");
  std::vector<ClusterT *> leafs;
  CT.getLeafIterator(leafs);
  int numInd = 0;
  for (auto i = 0; i < leafs.size(); ++i)
    numInd += (leafs[i])->get_indices().size();
  std::cout << leafs.size() << " " << numInd << "\n";
  for (auto level = 0; level < 14; ++level) {
    std::vector<Eigen::Matrix3d> bbvec;
    CT.get_BboxVector(&bbvec, level);
    FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
  }
  // ST.basisInfo();
  std::vector<std::vector<FMCA::IndexType>> tree;
  CT.exportTreeStructure(tree);
  for (auto i = 0; i < tree.size(); ++i) {
    int numInd = 0;
    for (auto j = 0; j < tree[i].size(); ++j)
      numInd += tree[i][j];
    std::cout << i << ") " << tree[i].size() << " " << numInd << "\n";
  }
  std::cout << "------------------------\n";
  T.tic();
  FMCA::BivariateCompressor<FMCA::SampletTree<ClusterT>> BC(ST, emptyFun());
  T.toc("set up compression pattern: ");
  auto pattern_triplets = BC.get_Pattern_triplets();
  plhs[0] = mxCreateDoubleMatrix(pattern_triplets.size(), 3, mxREAL);
  Eigen::Map<Eigen::MatrixXd> retU(mxGetPr(plhs[0]), pattern_triplets.size(),
                                   3);
  for (auto i = 0; i < pattern_triplets.size(); ++i)
    retU.row(i) << pattern_triplets[i].row(), pattern_triplets[i].col(),
        pattern_triplets[i].value();

  return;
}
