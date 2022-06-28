#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>

#include <FMCA/src/util/Macros.h>
#include <FMCA/src/util/MultiIndexSet.h>
#include <FMCA/src/Interpolators/LejaPoints.h>
#include <FMCA/src/Interpolators/WeightedTotalDegreeInterpolator.h>

#include "matrix.h"
#include "mex.h"

using Interpolator = FMCA::WeightedTotalDegreeInterpolator;
/** nlhs Number of expected output mxArrays
 *   plhs Array of pointers to the expected output mxArrays
 *   nrhs Number of input mxArrays
 *   prhs Array of pointers to the input mxArrays.
 *        Do not modify any prhs values in your MEX file.
 *        Changing the data in these read-only mxArrays can
 *        produce undesired side effects.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 3) {
    mexErrMsgIdAndTxt(
        "MATLAB:MXpolynomialInterpolator:nargin",
        "MXpolynomialInterpolator requires three arguments. Check help.");
  }
  // extract dimensions and data from Matlab
  const double *p_values = mxGetPr(prhs[0]);
  const int dimM = *(mxGetDimensions(prhs[0]));
  const int dimN = *(mxGetDimensions(prhs[0]) + 1);
  const double *weights = mxGetPr(prhs[1]);
  const FMCA::IndexType deg = std::round(*(mxGetPr(prhs[2])));
  std::vector<double> internal_weights(dimM);
  for (auto i = 0; i < dimM; ++i) internal_weights[i] = weights[i];
  std::cout << "points:   " << dimM << "x" << dimN << std::endl;
  std::cout << "weights:\n";
  for (auto &&it : internal_weights) std::cout << it << " ";
  std::cout << std::endl;
  std::cout << "max_degree: " << deg << std::endl;
  Eigen::Map<const Eigen::MatrixXd> interp_values(p_values, dimM, dimN);
  const Eigen::MatrixXd P = interp_values;
  Interpolator interp;
  interp.init(deg, internal_weights);

  // return everything to Matlab
  // Xis
  plhs[0] =
      mxCreateDoubleMatrix(interp.Xi().rows(), interp.Xi().cols(), mxREAL);
  // invV
  plhs[1] =
      mxCreateDoubleMatrix(interp.invV().rows(), interp.invV().cols(), mxREAL);
  // basis evaluation
  plhs[2] = mxCreateDoubleMatrix(interp.Xi().cols(), P.cols(), mxREAL);
  Eigen::Map<Eigen::MatrixXd> retXi(mxGetPr(plhs[0]), interp.Xi().rows(),
                                    interp.Xi().cols());
  Eigen::Map<Eigen::MatrixXd> retInvV(mxGetPr(plhs[1]), interp.invV().rows(),
                                      interp.invV().cols());
  Eigen::Map<Eigen::MatrixXd> retEval(mxGetPr(plhs[2]), interp.Xi().cols(),
                                      P.cols());
  retXi = interp.Xi();
  retInvV = interp.invV();
  for (auto i = 0; i < P.cols(); ++i)
    retEval.col(i) = interp.evalPolynomials(P.col(i));

  return;
}
