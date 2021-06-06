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
#include "FMCA/src/util/tictoc.hpp"
#include "matrix.h"
#include "mex.h"
#define NPTS 16384
//#define NPTS 8192
//#define NPTS 1800
//#define NPTS 1000
//#define NPTS 512
#define DIM 3
#define TEST_SAMPLET_TRANSFORM_
#define TEST_COMPRESSOR_

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
  if (nrhs != 1) {
    mexErrMsgIdAndTxt("MATLAB:MXmain:nargin", "MXmain zero. Check help.");
  }
  FMCA::IndexType npts = std::round(*(mxGetPr(prhs[0])));
  srand(0);
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, npts);
  tictoc T;
  T.tic();
  ClusterT CT(P);
  FMCA::SampletTree<ClusterT> ST(P, CT, 1);
  T.toc("set up ct: ");
  T.tic();
  FMCA::BivariateCompressor<FMCA::SampletTree<ClusterT>> BC(P, ST, Gaussian());
  T.toc("set up compression pattern: ");
  auto pattern_triplets = BC.get_Pattern_triplets();
  plhs[0] = mxCreateDoubleMatrix(pattern_triplets.size(), 3, mxREAL);
  Eigen::Map<Eigen::MatrixXd> retU(mxGetPr(plhs[0]), pattern_triplets.size(),
                                   3);
  for (auto i = 0; i < pattern_triplets.size(); ++i)
    retU.row(i) << pattern_triplets[i].row() + 1, pattern_triplets[i].col() + 1,
        pattern_triplets[i].value();

  return;
}
