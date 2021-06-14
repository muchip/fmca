#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <functional>
#include <iomanip>

#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"
#include "matrix.h"
#include "mex.h"

#define DIM 3
#define MPOLE_DEG 3

struct Gaussian {
  double operator()(const Eigen::Matrix<double, DIM, 1> &x,
                    const Eigen::Matrix<double, DIM, 1> &y) const {
    return exp(-4 * (x - y).norm());
  }
};

using ClusterT = FMCA::ClusterTree<double, DIM, 10>;

/** nlhs Number of expected output mxArrays
 *   plhs Array of pointers to the expected output mxArrays
 *   nrhs Number of input mxArrays
 *   prhs Array of pointers to the input mxArrays.
 *        Do not modify any prhs values in your MEX file.
 *        Changing the data in these read-only mxArrays can
 *        produce undesired side effects.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 2) {
    mexErrMsgIdAndTxt("MATLAB:MXmain:nargin",
                      "MXmain requires two arguments. Check help.");
  }
  FMCA::IndexType dtilde = std::round(*(mxGetPr(prhs[1])));
  std::string filename(mxArrayToString(prhs[0]));
  std::cout << "loading data: ";
  Eigen::MatrixXd B = readMatrix(filename);
  std::cout << "data size: ";
  std::cout << B.rows() << " " << B.cols() << std::endl;
  std::cout << "----------------------------------------------------\n";
  Eigen::MatrixXd P = B.transpose();

  tictoc T;
  T.tic();
  ClusterT CT(P);
  T.tic();
  FMCA::H2ClusterTree<ClusterT, MPOLE_DEG> H2CT(P, CT);
  T.toc("set up H2-cluster tree: ");
  FMCA::SampletTree<ClusterT> ST(P, CT, dtilde);
  T.toc("set up ct: ");
  std::vector<ClusterT *> leafs;
  CT.getLeafIterator(leafs);
  int numInd = 0;
  for (auto i = 0; i < leafs.size(); ++i)
    numInd += (leafs[i])->get_indices().size();
  for (auto level = 0; level < 14; ++level) {
    std::vector<Eigen::Matrix3d> bbvec;
    CT.get_BboxVector(&bbvec, level);
    FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
  }
  std::vector<Eigen::Matrix<double, DIM, 3u>> bbvec;
  CT.get_BboxVectorLeafs(&bbvec);
  FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
  FMCA::IO::plotPoints("points.vtk", P);
  T.tic();
  ST.computeMultiscaleClusterBases(H2CT);
  T.toc("set up time multiscale cluster bases");
  T.tic();
  FMCA::BivariateCompressorH2<FMCA::SampletTree<ClusterT>> BC(P, ST, Gaussian());
  T.toc("set up compressed matrixx: ");
  auto pattern_triplets = BC.get_Pattern_triplets();
  plhs[0] = mxCreateDoubleMatrix(pattern_triplets.size(), 3, mxREAL);
  Eigen::Map<Eigen::MatrixXd> retU(mxGetPr(plhs[0]), pattern_triplets.size(),
                                   3);
  for (auto i = 0; i < pattern_triplets.size(); ++i)
    retU.row(i) << pattern_triplets[i].row() + 1, pattern_triplets[i].col() + 1,
        pattern_triplets[i].value();

  return;
}
