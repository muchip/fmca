// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU General Public License version 3
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_UTIL_FORWARDDECLARATIONS_H_
#define FMCA_UTIL_FORWARDDECLARATIONS_H_

namespace FMCA {

template <typename Derived> struct ClusterTreeBase;

namespace internal {
template <typename Derived> struct ClusterTreeInitializer;
}

class ClusterTree;

template <typename ClusterTreeType> class SampletTree;

struct ClusterTreeNode;

struct SampletTreeNode;

} // namespace FMCA

#endif
