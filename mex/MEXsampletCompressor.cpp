#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include <FMCA/MatrixEvaluators>
#include <FMCA/Samplets>

#include "matrix.h"
#include "mex.h"

struct expKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-(x - y).norm());
  }
};

using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromMatrixEvaluator<Moments, expKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
/** nlhs Number of expected output mxArrays
 *   plhs Array of pointers to the expected output mxArrays
 *   nrhs Number of input mxArrays
 *   prhs Array of pointers to the input mxArrays.
 *        Do not modify any prhs values in your MEX file.
 *        Changing the data in these read-only mxArrays can
 *        produce undesired side effects.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 4) {
    mexErrMsgIdAndTxt(
        "MATLAB:MXsampletCompressor:nargin",
        "MXsampletCompressor requires four arguments. Check help.");
  }
  // extract dimensions and data from Matlab
  const double *p_values = mxGetPr(prhs[0]);
  const int dimM = *(mxGetDimensions(prhs[0]));
  const int dimN = *(mxGetDimensions(prhs[0]) + 1);
  const FMCA::IndexType dtilde = std::round(*(mxGetPr(prhs[1])));
  const double eta = *(mxGetPr(prhs[2]));
  const double threshold = *(mxGetPr(prhs[3]));

  std::cout << "points:   " << dimM << "x" << dimN << std::endl;
  std::cout << "dtilde:   " << dtilde << std::endl;
  std::cout << "leafsize: " << leafsize << std::endl;

  // init samplet tree
  Eigen::Map<const Eigen::MatrixXd> interp_values(p_values, dimM, dimN);
  Eigen::MatrixXd P = interp_values;
  FMCA::SampletTreeQR ST(P, leafsize, dtilde);
  std::vector<Eigen::Triplet<double>> trafo_triplets;
  trafo_triplets = ST.transformationMatrixTriplets();

  // return everything to Matlab
  plhs[0] = mxCreateDoubleMatrix(trafo_triplets.size(), 3, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(P.cols(), 1, mxREAL);
  Eigen::Map<Eigen::MatrixXd> retT(mxGetPr(plhs[0]), trafo_triplets.size(), 3);
  Eigen::Map<Eigen::MatrixXd> retI(mxGetPr(plhs[1]), P.cols(), 1);
  for (auto i = 0; i < trafo_triplets.size(); ++i)
    retT.row(i) << trafo_triplets[i].row() + 1, trafo_triplets[i].col() + 1,
        trafo_triplets[i].value();
  for (auto i = 0; i < ST.indices().size(); ++i)
    retI(i) = ST.indices()[i] + 1;
  return;
}
