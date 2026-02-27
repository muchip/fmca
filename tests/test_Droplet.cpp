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
#include "../FMCA/src/util/CombiIndexSet.h"
#include "../FMCA/src/util/Droplet.h"
#include "../FMCA/src/util/MultiIndexSet.h"
#include "../FMCA/src/util/Tictoc.h"

typedef std::set<std::vector<FMCA::Index>,
                 FMCA::FMCA_Compare<std::vector<FMCA::Index>>>
    multi_index_set;

typedef std::map<std::vector<FMCA::Index>, std::ptrdiff_t,
                 FMCA::FMCA_Compare<std::vector<FMCA::Index>>>
    combi_index_set;

int main() {
  FMCA::Tictoc T;
  multi_index_set myset;
  // test tensor product droplet
  {
    std::vector<FMCA::Index> n(4);
    std::vector<FMCA::Index> index(4);
    n[0] = 10;
    n[1] = 5;
    n[2] = 3;
    n[3] = 0;

    FMCA::tensorProductDroplet(n, myset);
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
  {
    std::vector<FMCA::Index> index(4);
    FMCA::Vector w(4);
    FMCA::Index q = 5;
    w << 4., 1. / 3, 2., 1;
    FMCA::weightedTotalDegreeDroplet(w, q, myset);
    for (FMCA::Index i0 = 0; i0 <= q / w[0] + 1; ++i0)
      for (FMCA::Index i1 = 0; i1 <= q / w[1] + 1; ++i1)
        for (FMCA::Index i2 = 0; i2 <= q / w[2] + 1; ++i2)
          for (FMCA::Index i3 = 0; i3 <= q / w[3] + 1; ++i3) {
            index[0] = i0;
            index[1] = i1;
            index[2] = i2;
            index[3] = i3;
            FMCA::Scalar scap = 0;
            for (FMCA::Index j = 0; j < 4; ++j)
              scap += FMCA::Scalar(index[j]) * w[j];
            if (scap <= q)
              if (myset.find(index) == myset.end()) {
                std::cerr << "index not present in the wtdindex set"
                          << std::endl;
                return 1;
              }
          }
  }

  {
    FMCA::Index dim = 20;
    FMCA::Scalar q = 40;
    std::vector<FMCA::Scalar> w(dim);
    for (FMCA::Index i = 0; i < dim; ++i) w[i] = (i + 1);
    T.tic();
    FMCA::MultiIndexSet<FMCA::WeightedTotalDegree> wcset(dim, q, w);
    T.toc("WTD: ");
    T.tic();
    FMCA::weightedTotalDegreeDroplet(w, q, myset);
    T.toc("WTD droplet: ");
    std::cout << "set size: " << myset.size() << std::endl;
    assert(wcset.index_set().size() == myset.size() && "size mismatch");
    for (const auto &it : wcset.index_set()) {
      assert(myset.find(it) != myset.end() && "missing index");
    }
  }

  {
    combi_index_set myset;
    FMCA::Index dim = 20;
    FMCA::Scalar q = 20;
    std::vector<FMCA::Scalar> w(dim);
    for (FMCA::Index i = 0; i < dim; ++i) w[i] = (i + 1);
    T.tic();
    FMCA::CombiIndexSet<FMCA::WeightedTotalDegree> wcset(dim, q, w);
    std::cout << wcset.index_set().size() << std::endl;
    T.toc("WTD combi: ");
    FMCA::weightedTotalDegreeCombiDroplet(w, q, myset);
    T.toc("WTD combi droplet: ");
    std::cout << "set size: " << myset.size() << std::endl;
    assert(wcset.index_set().size() == myset.size() && "size mismatch");
    for (const auto &it : wcset.index_set()) {
      auto it2 = myset.find(it.first);
      assert(it2 != myset.end() && "missing index");
      assert(it2->second == it.second && "wrong combination weight");
    }
  }

  return 0;
}
