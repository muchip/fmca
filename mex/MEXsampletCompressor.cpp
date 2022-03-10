#define FMCA_CLUSTERSET_
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/Sparse>
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
  if (nrhs != 5) {
    mexErrMsgIdAndTxt(
        "MATLAB:MXsampletCompressor:nargin",
        "MXsampletCompressor requires five arguments. Check help.");
  }
  // extract dimensions and data from Matlab
  const double *p_values = mxGetPr(prhs[0]);
  const int dimM = *(mxGetDimensions(prhs[0]));
  const int dimN = *(mxGetDimensions(prhs[0]) + 1);
  const FMCA::IndexType dtilde = std::round(*(mxGetPr(prhs[1])));
  const double eta = *(mxGetPr(prhs[2]));
  const FMCA::IndexType mp_deg = std::round(*(mxGetPr(prhs[3])));
  const double threshold = *(mxGetPr(prhs[4]));
  const auto function = expKernel();

  std::cout << "points:   " << dimM << "x" << dimN << std::endl;
  std::cout << "dtilde:   " << dtilde << std::endl;
  // init samplet tree
  Eigen::Map<const Eigen::MatrixXd> interp_values(p_values, dimM, dimN);
  const Eigen::MatrixXd P = interp_values;
  const Moments mom(P, mp_deg);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, P);
  FMCA::symmetric_compressor_impl<H2SampletTree> comp;
  comp.compress(hst, mat_eval, eta, threshold);
  const auto &trips = comp.pattern_triplets();
  // return everything to Matlab
  plhs[0] = mxCreateDoubleMatrix(trips.size(), 3, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(P.cols(), 1, mxREAL);
  Eigen::Map<Eigen::MatrixXd> retK(mxGetPr(plhs[0]), trips.size(), 3);
  Eigen::Map<Eigen::MatrixXd> retI(mxGetPr(plhs[1]), P.cols(), 1);
  for (auto i = 0; i < trips.size(); ++i)
    retK.row(i) << trips[i].row() + 1, trips[i].col() + 1, trips[i].value();
  for (auto i = 0; i < hst.indices().size(); ++i)
    retI(i) = hst.indices()[i] + 1;
  return;
}
