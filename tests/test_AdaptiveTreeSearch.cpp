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
#include "../FMCA/Samplets"

#define DIM 2
#define NPTS 1000000

using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;

int main() {
  const FMCA::Index dtilde = 4;
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  FMCA::Vector data(P.cols());
  for (FMCA::Index i = 0; i < P.cols(); ++i)
    data(i) = std::exp(-P.col(i).norm());
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

  return 0;
}
