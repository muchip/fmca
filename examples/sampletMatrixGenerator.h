// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.

#ifndef FMCA_SAMPLETMATRIXGENERATOR_
#define FMCA_SAMPLETMATRIXGENERATOR_

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
struct SampletCRS {

  std::vector<Eigen::Triplet<double>> toTriplets() {
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(a.size());
    unsigned int n = ia.size() - 1;
    for (auto i = 0; i < n; ++i)
      for (auto j = ia[i]; j < ia[i + 1]; ++j)
        trips.push_back(Eigen::Triplet<double>(i, ja[j], a[j]));
    return trips;
  }

  size_t nnz() { return a.size(); }
  double pnnz() {
    return double(a.size()) / (ia.size() - 1) / (ia.size() - 1) * 100.;
  }
  //////////////////////////////////////////////////////////////////////////////
  std::vector<double> a;
  std::vector<int> ia;
  std::vector<int> ja;
  double unif_const;
  double comp_time;
  double comp_err;
};

SampletCRS sampletMatrixGenerator(const Eigen::MatrixXd &P,
                                  const unsigned int mp_deg,
                                  const unsigned int dtilde, const double eta,
                                  double threshold, const double ridgep);

#endif
