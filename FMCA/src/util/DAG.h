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
#ifndef FMCA_UTIL_COMPRESSORDAG_H_
#define FMCA_UTIL_COMPRESSORDAG_H_

namespace FMCA {
namespace internal {

enum RecyclingStrategy { Rows, Cols, Leaf };

template <typename Derived, typename PayloadType>
struct CompressorDAG {
  const Derived *pr = nullptr;
  const Derived *pc = nullptr;
  std::vector<CompressorDAG *> row_sons;
  std::vector<CompressorDAG *> col_sons;
  CompressorDag *row_dad = nullptr;
  CompressorDag *col_dad = nullptr;
  RecyclingStrategy strategy = Leaf;

  PayloadType block;

  std::atomic<Index> deps_remaining{0};
  std::atomic<Index> consumers_remaining{0};
  std::atomic<bool> claimed{false};
};

}  // namespace internal
}  // namespace FMCA
#endif
