#define USE_QR_CONSTRUCTION_
#define FMCA_CLUSTERSET_

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

#define MPOLE_DEG 5

struct Gaussian {
  double operator()(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const {
    // return (1 + sqrt(3) * 20 * (x - y).norm()) *
    //        exp(-20 * sqrt(3) * (x - y).norm());
    return exp(-20 * (x - y).norm());
  }
};

using ClusterT1 = FMCA::ClusterTree<double, 1, 10, MPOLE_DEG>;
using ClusterT2 = FMCA::ClusterTree<double, 2, 10, MPOLE_DEG>;
using ClusterT3 = FMCA::ClusterTree<double, 3, 10, MPOLE_DEG>;
using ClusterT4 = FMCA::ClusterTree<double, 4, 10, MPOLE_DEG>;
using ClusterT5 = FMCA::ClusterTree<double, 5, 10, MPOLE_DEG>;
using H2ClusterT1 = FMCA::H2ClusterTree<ClusterT1, MPOLE_DEG>;
using H2ClusterT2 = FMCA::H2ClusterTree<ClusterT2, MPOLE_DEG>;
using H2ClusterT3 = FMCA::H2ClusterTree<ClusterT3, MPOLE_DEG>;
using H2ClusterT4 = FMCA::H2ClusterTree<ClusterT4, MPOLE_DEG>;
using H2ClusterT5 = FMCA::H2ClusterTree<ClusterT5, MPOLE_DEG>;

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
  auto p_values = mxGetPr(prhs[0]);
  int dimM = *(mxGetDimensions(prhs[0]));
  int dimN = *(mxGetDimensions(prhs[0]) + 1);
  std::cout << "M: " << dimM << " N: " << dimN << std::endl;
  Eigen::Map<Eigen::MatrixXd> interp_values(p_values, dimM, dimN);
  Eigen::MatrixXd P = interp_values.transpose();
#ifdef USE_QR_CONSTRUCTION_
  std::cout << "using QR construction for samplet basis\n";
#endif
  std::vector<Eigen::Triplet<double>> pattern_triplets;
  std::vector<Eigen::Triplet<double>> trafo_triplets;
  std::vector<FMCA::IndexType> idcs;
  switch (dimN) {
    case 1: {
      ClusterT1 CT(P);
      H2ClusterT1 H2CT(P, CT);
      FMCA::SampletTree<ClusterT1> ST(P, CT, dtilde);
      ST.computeMultiscaleClusterBases(H2CT);
      FMCA::BivariateCompressorH2<FMCA::SampletTree<ClusterT1>> BC;
      BC.set_eta(0.8);
      BC.set_threshold(0);
      BC.init(P, ST, Gaussian());
      pattern_triplets = BC.get_Pattern_triplets();
      trafo_triplets = ST.get_transformationMatrix();
      idcs = CT.get_indices();
      break;
    }
    case 2: {
      ClusterT2 CT(P);
      H2ClusterT2 H2CT(P, CT);
      FMCA::SampletTree<ClusterT2> ST(P, CT, dtilde);
      ST.computeMultiscaleClusterBases(H2CT);
      FMCA::BivariateCompressorH2<FMCA::SampletTree<ClusterT2>> BC;
      BC.set_eta(0.8);
      BC.set_threshold(0);
      BC.init(P, ST, Gaussian());
      pattern_triplets = BC.get_Pattern_triplets();
      trafo_triplets = ST.get_transformationMatrix();
      idcs = CT.get_indices();
      break;
    }
    case 3: {
      ClusterT3 CT(P);
      H2ClusterT3 H2CT(P, CT);
      FMCA::SampletTree<ClusterT3> ST(P, CT, dtilde);
      ST.computeMultiscaleClusterBases(H2CT);
      FMCA::BivariateCompressorH2<FMCA::SampletTree<ClusterT3>> BC;
      BC.set_eta(0.8);
      BC.set_threshold(0);
      BC.init(P, ST, Gaussian());
      pattern_triplets = BC.get_Pattern_triplets();
      trafo_triplets = ST.get_transformationMatrix();
      idcs = CT.get_indices();
      break;
    }
    default:
      std::cout << "THIS IS NOT IMPLEMENTED!!!\n";
  }
  plhs[0] = mxCreateDoubleMatrix(pattern_triplets.size(), 3, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(trafo_triplets.size(), 3, mxREAL);
  plhs[2] = mxCreateDoubleMatrix(P.cols(), 1, mxREAL);

  Eigen::Map<Eigen::MatrixXd> retU(mxGetPr(plhs[0]), pattern_triplets.size(),
                                   3);
  Eigen::Map<Eigen::MatrixXd> retQ(mxGetPr(plhs[1]), trafo_triplets.size(), 3);

  Eigen::Map<Eigen::MatrixXd> retI(mxGetPr(plhs[2]), P.cols(), 1);

  for (auto i = 0; i < pattern_triplets.size(); ++i)
    retU.row(i) << pattern_triplets[i].row() + 1, pattern_triplets[i].col() + 1,
        pattern_triplets[i].value();

  for (auto i = 0; i < trafo_triplets.size(); ++i)
    retQ.row(i) << trafo_triplets[i].row() + 1, trafo_triplets[i].col() + 1,
        trafo_triplets[i].value();

  for (auto i = 0; i < idcs.size(); ++i) retI(i) = idcs[i] + 1;

  return;
}
