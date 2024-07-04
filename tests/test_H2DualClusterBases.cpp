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
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/src/H2Matrix/compute_dual_cluster_bases_impl.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 100000
#define DIM 3
#define MPOLE_DEG 3

using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using MatrixEvaluatorUS =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, 2 * NPTS);

  const Moments mom(P, MPOLE_DEG);

  std::cout
      << "Cluster splitter:             "
      << FMCA::internal::traits<FMCA::ClusterTree>::Splitter::splitterName()
      << std::endl;
  T.tic();
  H2ClusterTree ct(mom, 0, P);
  T.toc("H2 cluster tree:");
  FMCA::internal::compute_cluster_bases_impl::check_transfer_matrices(ct, mom);
  std::vector<FMCA::Matrix> VTVs;
  T.tic();
  FMCA::internal::compute_dual_cluster_bases_impl::compute(ct, &VTVs);
  T.toc("H2 dual cluster bases:");
  FMCA::internal::compute_dual_cluster_bases_impl::check_dual_cluster_bases(
      ct, VTVs);

  return 0;
}
