// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#include <Eigen/Dense>
#include <iostream>
#include <set>

#include "../FMCA/src/util/Droplet.h"
#include "../FMCA/src/util/MultiIndexSet.h"
#include "../FMCA/src/util/Tictoc.h"

typedef std::set<std::vector<FMCA::Index>,
                 FMCA::FMCA_Compare<std::vector<FMCA::Index>>>
    multi_index_set;

int main() {
  multi_index_set myset;
  // test tensor product droplet
  {
    std::vector<FMCA::Index> n(4);
    std::vector<FMCA::Index> index(4);
    n[0] = 10;
    n[1] = 5;
    n[2] = 3;
    n[3] = 0;
    FMCA::TensorProductDroplet(n, myset);
    for (FMCA::Index i0 = 0; i0 <= n[0]; ++i0)
      for (FMCA::Index i1 = 0; i1 <= n[1]; ++i1)
        for (FMCA::Index i2 = 0; i2 <= n[2]; ++i2)
          for (FMCA::Index i3 = 0; i3 <= n[3]; ++i3) {
            index[0] = i0;
            index[1] = i1;
            index[2] = i2;
            index[3] = i3;
            if (myset.find(index) == myset.end()) {
              std::cerr << "index not present in the tpindex set" << std::endl;
              return 1;
            }
          }
  }
  return 0;
}
