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
#ifndef FMCA_CLUSTERING_TREELEVELMAPPER_H_
#define FMCA_CLUSTERING_TREELEVELMAPPER_H_

#include <vector>

#include "../util/Macros.h"
#include "../util/TreeBase.h"

namespace FMCA {

template <typename Derived>
void treeLevelMapper(const ClusterTreeBase<Derived> &ct,
                     std::vector<Index> *s_lvl, std::vector<Index> *id_map,
                     std::vector<const Derived *> *tree_map = nullptr) {
  Index k = 0;
  Index l = 0;
  Index maxl = 0;
  Index max_id = 0;
  Index cur_level = -1;
  // first determine maximum level and maximum block_id of the tree
  for (const auto &it : ct) {
    maxl = maxl > it.level() ? maxl : it.level();
    max_id = max_id > it.block_id() ? max_id : it.block_id();
  }
  // s_lvl contains the offsets for each level
  s_lvl->resize(maxl + 2);
  id_map->resize(max_id + 1);

  // the tree iterator does a levelwise traversal of the tree
  // whenever a new level starts, it starting position is written to s_lvl
  // furthermore, we have a unique mapping assigning an index to a block
  if (tree_map) {
    tree_map->resize(max_id + 1);
    for (const auto &it : ct) {
      (*id_map)[it.block_id()] = k;
      (*tree_map)[k] = std::addressof(it);
      if (it.level() != cur_level) {
        cur_level = it.level();
        (*s_lvl)[l] = k;
        ++l;
      }
      ++k;
    }
  } else {
    for (const auto &it : ct) {
      (*id_map)[it.block_id()] = k;
      if (it.level() != cur_level) {
        cur_level = it.level();
        (*s_lvl)[l] = k;
        ++l;
      }
      ++k;
    }
  }
  (*s_lvl)[l] = k;

  return;
}

}  // namespace FMCA
#endif
