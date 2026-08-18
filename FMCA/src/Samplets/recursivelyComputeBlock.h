// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_SAMPLETS_RECURSIVELYCOMPUTEBLOCK_H_
#define FMCA_SAMPLETS_RECURSIVELYCOMPUTEBLOCK_H_

/**
 *  \brief recursively computes for a given pair of row and column
 *clusters the four blocks [A^PhiPhi, A^PhiSigma; A^SigmaPhi,
 *A^SigmaSigma]
 **/

#include "../util/Macros.h"

namespace FMCA {
namespace internal {

#ifdef FMCA_VERBOSE
static std::array<std::size_t, 5> recursivelyComputeBlockCounters;
#endif

template <typename H2STreeType, typename EntryGenerator,
          typename ClusterComparison = CompareCluster>
Matrix recursivelyComputeBlock(const H2STreeType &TR, const H2STreeType &TC,
                               const EntryGenerator &e_gen, const Scalar eta) {
  Matrix buf(0, 0);
  Index r_offset = 0;
  Index c_offset = 0;
  // check for admissibility
  if (ClusterComparison::compare(TR, TC, eta) == LowRank) {
    e_gen.interpolate_kernel(TR, TC, &buf);
#ifdef FMCA_VERBOSE
    ++recursivelyComputeBlockCounters[4];
#endif
    return TR.V().transpose() * buf * TC.V();
  } else {
    const char the_case = 2 * (!TR.nSons()) + !TC.nSons();
    switch (the_case) {
      case 3:
        // both are leafs: compute the block and return
        e_gen.compute_dense_block(TR, TC, &buf);
#ifdef FMCA_VERBOSE
        ++recursivelyComputeBlockCounters[3];
#endif
        return TR.Q().transpose() * buf * TC.Q();
      case 2:
        // the row cluster is a leaf cluster: recursion on the col cluster
        buf.resize(TR.Q().cols(), TC.Q().rows());
        c_offset = 0;
        for (Index j = 0; j < TC.nSons(); ++j) {
          const Index nscalfs = TC.sons(j).nscalfs();
          const Matrix ret =
              recursivelyComputeBlock<H2STreeType, EntryGenerator,
                                      ClusterComparison>(TR, TC.sons(j), e_gen,
                                                         eta);
          buf.middleCols(c_offset, nscalfs) = ret.leftCols(nscalfs);
          c_offset += nscalfs;
        }
#ifdef FMCA_VERBOSE
        ++recursivelyComputeBlockCounters[2];
#endif
        return buf * TC.Q();
      case 1:
        // the col cluster is a leaf cluster: recursion on the row cluster
        buf.resize(TR.Q().rows(), TC.Q().cols());
        r_offset = 0;
        for (Index i = 0; i < TR.nSons(); ++i) {
          const Index nscalfs = TR.sons(i).nscalfs();
          const Matrix ret =
              recursivelyComputeBlock<H2STreeType, EntryGenerator,
                                      ClusterComparison>(TR.sons(i), TC, e_gen,
                                                         eta);
          buf.middleRows(r_offset, nscalfs) = ret.topRows(nscalfs);
          r_offset += nscalfs;
        }
#ifdef FMCA_VERBOSE
        ++recursivelyComputeBlockCounters[1];
#endif
        return TR.Q().transpose() * buf;
      case 0:
        // neither is a leaf, let recursion handle this
        buf.resize(TR.Q().rows(), TC.Q().cols());
        r_offset = 0;
        for (Index i = 0; i < TR.nSons(); ++i) {
          Matrix buf2(TR.sons(i).Q().cols(), TC.Q().rows());
          c_offset = 0;
          const Index r_nscalfs = TR.sons(i).nscalfs();
          for (Index j = 0; j < TC.nSons(); ++j) {
            const Index c_nscalfs = TC.sons(j).nscalfs();
            const Matrix ret =
                recursivelyComputeBlock<H2STreeType, EntryGenerator,
                                        ClusterComparison>(
                    TR.sons(i), TC.sons(j), e_gen, eta);
            buf2.middleCols(c_offset, c_nscalfs) = ret.leftCols(c_nscalfs);
            c_offset += c_nscalfs;
          }
          buf.middleRows(r_offset, r_nscalfs).noalias() =
              buf2.topRows(r_nscalfs) * TC.Q();
          r_offset += r_nscalfs;
        }
#ifdef FMCA_VERBOSE
        ++recursivelyComputeBlockCounters[0];
#endif
        return TR.Q().transpose() * buf;
    }
  }
  return Matrix(0, 0);
}
}  // namespace internal
}  // namespace FMCA
#endif
