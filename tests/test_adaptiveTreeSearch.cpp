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

#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/adaptiveTreeSearch.h"
#include "../FMCA/src/util/IO.h"

#define DIM 2
#define NPTS 10000

using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;

int main() {
  const FMCA::Index dtilde = 4;
  const FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, NPTS);
  FMCA::Vector data(P.cols());
  for (FMCA::Index i = 0; i < P.cols(); ++i)
    // data(i) = std::exp(-P.col(i).norm());
    data(i) = (0.5 * std::tanh(2000 * (P(0, i) + P(1, i) - 1)));
  const SampletMoments samp_mom(P, dtilde - 1);
  const SampletTree st(samp_mom, 0, P);

  // sort data with respect to the cluster tree order
  FMCA::Vector sdata = st.toClusterOrder(data);
  FMCA::Vector tdata = st.sampletTransform(sdata);
  FMCA::Scalar norm2 = tdata.squaredNorm();
  std::vector<const SampletTree *> adaptive_tree =
      adaptiveTreeSearch(st, tdata, 1e-6 * norm2);
  const FMCA::Index nclusters = std::distance(st.begin(), st.end());

  FMCA::Vector thres_tdata = tdata;
  thres_tdata.setZero();
  FMCA::Index nnz = 0;
  for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    if (adaptive_tree[i] != nullptr) {
      const SampletTree &node = *(adaptive_tree[i]);
      const FMCA::Index ndist =
          node.is_root() ? node.Q().cols() : node.nsamplets();
      thres_tdata.segment(node.start_index(), ndist) =
          tdata.segment(node.start_index(), ndist);
      nnz += ndist;
    }
  }
  std::cout << "active coefficients: " << nnz << " / " << NPTS << std::endl;
  std::cout << "tree error: " << (thres_tdata - tdata).norm() / tdata.norm()
            << std::endl;

  std::vector<bool> check_id(NPTS);
  for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    if (adaptive_tree[i] != nullptr) {
      const SampletTree &node = *(adaptive_tree[i]);
      if (!node.nSons() || adaptive_tree[node.sons(0).block_id()] == nullptr)
        for (FMCA::Index j = 0; j < node.block_size(); ++j)
          check_id[node.indices()[j]] = true;
    }
  }
  for (FMCA::Index i = 0; i < check_id.size(); ++i) {
    if (!check_id[i]) {
      std::cerr << "missing index in adaptive tree";
      return 1;
    }
  }

  // Collect bounding boxes and colors for active leaves
  std::vector<FMCA::Matrix> active_bbvec;
  std::vector<FMCA::Scalar> active_colr;
  FMCA::Vector point_colr(NPTS);
  point_colr.setZero(); 

  for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    if (adaptive_tree[i] != nullptr) {
      const SampletTree &node = *(adaptive_tree[i]);
      bool is_adaptive_leaf =
          !node.nSons() ||
          (node.nSons() && adaptive_tree[node.sons(0).block_id()] == nullptr);
      if (is_adaptive_leaf) {
        const FMCA::Scalar rdm = std::rand() % 256;
        for (FMCA::Index j = 0; j < node.block_size(); ++j) {
          point_colr(node.indices()[j]) = rdm;
        }
        if (node.block_size()) {
          active_bbvec.push_back(node.bb());
          active_colr.push_back(rdm);
        }
      }
    }
  }

  FMCA::Matrix P3D(3, NPTS);
  P3D.setZero();
  P3D.topRows(2) = P;

  FMCA::IO::plotPointsColor("adaptive_clusters.vtk", P3D, point_colr);
  FMCA::IO::plotBoxes2D("adaptive_boxes.vtk", active_bbvec, active_colr);

  std::cout << "Number of active leaves: " << active_bbvec.size() << std::endl;
  return 0;
}