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

#define DIM 1
#define MPOLE_DEG 3

struct Gaussian {
  double operator()(const Eigen::Matrix<double, DIM, 1> &x,
                    const Eigen::Matrix<double, DIM, 1> &y) const {
    return (1 + sqrt(3) * 20 * (x - y).norm()) *
           exp(-20 * sqrt(3) * (x - y).norm());
  }
};

using ClusterT = FMCA::ClusterTree<double, DIM, 10, MPOLE_DEG>;
using H2ClusterT = FMCA::H2ClusterTree<ClusterT, MPOLE_DEG>;

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
  std::cout << std::string(60, '-') << std::endl;
  Eigen::MatrixXd P = B.transpose();

  tictoc T;
  T.tic();
  ClusterT CT(P);
  T.tic();
  H2ClusterT H2CT(P, CT);
  T.toc("set up H2-cluster tree: ");
  FMCA::SampletTree<ClusterT> ST(P, CT, dtilde);
  T.toc("set up ct: ");
  T.tic();
  ST.computeMultiscaleClusterBases(H2CT);
  T.toc("set up time multiscale cluster bases");
  T.tic();
  FMCA::BivariateCompressorH2<FMCA::SampletTree<ClusterT>> BC;
  BC.set_eta(0.4);
  BC.set_threshold(0);
  BC.init(P, ST, Gaussian());
  T.toc("set up Samplet compressed matrix: ");
  auto pattern_triplets = BC.get_Pattern_triplets();
  auto trafo_triplets = ST.get_transformationMatrix();
  plhs[0] = mxCreateDoubleMatrix(pattern_triplets.size(), 3, mxREAL);
  plhs[2] = mxCreateDoubleMatrix(trafo_triplets.size(), 3, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(P.cols(), 1, mxREAL);

  Eigen::Map<Eigen::MatrixXd> retU(mxGetPr(plhs[0]), pattern_triplets.size(),
                                   3);
  Eigen::Map<Eigen::MatrixXd> retQ(mxGetPr(plhs[2]), trafo_triplets.size(), 3);
  Eigen::Map<Eigen::MatrixXd> retI(mxGetPr(plhs[1]), P.cols(), 1);

  for (auto i = 0; i < pattern_triplets.size(); ++i)
    retU.row(i) << pattern_triplets[i].row() + 1, pattern_triplets[i].col() + 1,
        pattern_triplets[i].value();

  for (auto i = 0; i < trafo_triplets.size(); ++i)
    retQ.row(i) << trafo_triplets[i].row() + 1, trafo_triplets[i].col() + 1,
        trafo_triplets[i].value();

  auto idcs = CT.get_indices();
  for (auto i = 0; i < idcs.size(); ++i) retI(i) = idcs[i] + 1;

  return;
}
