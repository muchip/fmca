#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "MultiIndexSet.h"
#include "matrix.h"
#include "mex.h"

/** nlhs Number of expected output mxArrays
 *   plhs Array of pointers to the expected output mxArrays
 *   nrhs Number of input mxArrays
 *   prhs Array of pointers to the input mxArrays.
 *        Do not modify any prhs values in your MEX file.
 *        Changing the data in these read-only mxArrays can
 *        produce undesired side effects.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  const unsigned int dim = std::round(*(mxGetPr(prhs[0])));
  const unsigned int q = std::round(*(mxGetPr(prhs[1])));

  MultiIndexSet<std::vector<int>> Xset;

  Xset.init(dim, q);

  const auto &X = Xset.get_MultiIndexSet();

  plhs[0] = mxCreateDoubleMatrix(dim, X.size(), mxREAL);
  double *p = mxGetPr(plhs[0]);
  int i = 0;
  for (const auto &ind : X) {
    for (auto j = 0; j < ind.size(); ++j)
      p[j + i * dim] = ind[j];
    ++i;
  }

  return;
}
