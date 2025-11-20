// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#include "../FMCA/src/util/Alpha.h"
#include "../FMCA/src/util/Tictoc.h"

int main() {
  FMCA::Tictoc T;
  FMCA::iVector nds(4);
  nds << 2, 4, 3, 5;
  T.tic();
  FMCA::Alpha alpha(nds);
  T.toc("computed multi mapping");
  assert(alpha.n() == nds.prod() && "dimension mismatch");

  FMCA::Index n = alpha.n();
  FMCA::iMatrix indices(4, n);
  std::cout << "testing " << n << " multi indices" << std::endl;
  T.tic();
  std::cout << "multi indices" << std::endl;
  for (FMCA::Index i = 0; i < n; ++i) {
    indices.col(i) = alpha.toMultiIndex<FMCA::iVector>(i);
    assert(alpha.toScalarIndex<FMCA::iVector>(indices.col(i)) == i &&
           "multi index mismatch");
    std::cout << i << " is " << indices.col(i).transpose() << std::endl;
  }
  const FMCA::Index dim = 3;
  std::cout << "matricization with respect to dimension " << dim << std::endl;
  for (FMCA::Index i = 0; i < alpha.nds()(dim); ++i) {
    for (FMCA::Index j = 0; j < alpha.n() / alpha.nds()(dim); ++j)
      std::cout << alpha.matricize(dim, i, j) << " ";
    std::cout << std::endl;
  }

  T.toc("elapsed time:");
  return 0;
}
