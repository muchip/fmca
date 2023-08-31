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
#ifndef FMCA_UTIL_FORWARDDECLARATIONS_H_
#define FMCA_UTIL_FORWARDDECLARATIONS_H_

namespace FMCA {

template <typename Derived>
struct ClusterTreeBase;

template <typename Derived>
struct H2ClusterTreeBase;

template <typename Derived>
struct H2SampletTreeBase;

template <typename Derived>
struct H2MatrixBase;

namespace internal {
template <typename Derived>
struct ClusterTreeInitializer;
}

class ClusterTree;

class MortonClusterTree;

template <typename ClusterTreeType>
class SampletTree;

template <typename ClusterTreeType>
class H2ClusterTree;

template <typename ClusterTreeType>
class H2SampletTree;

template <typename Derived>
struct H2Matrix;

struct ClusterTreeNode;

struct SampletTreeNode;

struct H2ClusterTreeNode;

struct H2SampletTreeNode;

template <typename ClusterTreeType>
struct H2MatrixNode;

}  // namespace FMCA

#endif
