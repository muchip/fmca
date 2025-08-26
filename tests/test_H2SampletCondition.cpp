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
// #define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS (1 << 10)
#define DIM 1

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("MaternNu", 1., 1., 1.5);
  const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  const FMCA::Scalar threshold = 1e-8;
  const FMCA::Scalar eta = 0.5;

  const FMCA::Index dtilde = 4;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const Moments mom(P, mpole_deg);
  const MatrixEvaluator mat_eval(mom, function);
  std::cout << "dtilde:                       " << dtilde << std::endl;
  std::cout << "mpole_deg:                    " << mpole_deg << std::endl;
  std::cout << "eta:                          " << eta << std::endl;
  const SampletMoments samp_mom(P, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.tic();
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Scomp;
  Scomp.init(hst, eta, threshold);
  T.toc("planner:                     ");
  T.tic();
  Scomp.compress(mat_eval);
  T.toc("compressor:                  ");
  T.tic();
  const auto &trips = Scomp.triplets();
  T.toc("triplets:                    ");
  std::cout << "anz:                          "
            << std::round(trips.size() / FMCA::Scalar(NPTS)) << std::endl;
  FMCA::Vector x(NPTS), y1(NPTS), y2(NPTS);
  FMCA::Scalar err = 0;
  FMCA::Scalar nrm = 0;
  for (auto i = 0; i < 10; ++i) {
    FMCA::Index index = rand() % P.cols();
    x.setZero();
    x(index) = 1;
    FMCA::Vector col = function.eval(P, P.col(hst.indices()[index]));
    y1 = col(Eigen::Map<const FMCA::iVector>(hst.indices(), hst.block_size()));
    x = hst.sampletTransform(x);
    y2.setZero();
    for (const auto &i : trips) {
      y2(i.row()) += i.value() * x(i.col());
      if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
    }
    y2 = hst.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  err = sqrt(err / nrm);
  std::cout << "compression error:            " << err << std::endl
            << std::flush;
  std::vector<int> level_mapper;
  FMCA::Index max_level = 0;
  // assign vector start_index to each wavelet cluster
  double geo_diam = hst.bb().col(2).norm();
  for (auto &&it : hst) {
    const double diam = it.bb().col(2).norm();
    const int level = it.level();
    max_level = max_level < level ? level : max_level;
    if (it.is_root())
      for (auto i = 0; i < it.nscalfs(); ++i) level_mapper.push_back(-1);
    for (auto i = 0; i < it.derived().nsamplets(); ++i)
      level_mapper.push_back(level);
  }
  std::vector<std::vector<FMCA::Index>> matrix_mapper(max_level + 2);
  std::vector<FMCA::Index> min_index(max_level + 2, -1);
  std::vector<FMCA::Index> max_index(max_level + 2, 0);

  for (FMCA::Index i = 0; i < trips.size(); ++i) {
    const auto &t = trips[i];
    if (level_mapper[t.row()] == level_mapper[t.col()]) {
      matrix_mapper[level_mapper[t.row()] + 1].push_back(i);
      min_index[level_mapper[t.row()] + 1] =
          min_index[level_mapper[t.row()] + 1] > t.row()
              ? t.row()
              : min_index[level_mapper[t.row()] + 1];
      min_index[level_mapper[t.row()] + 1] =
          min_index[level_mapper[t.row()] + 1] > t.col()
              ? t.col()
              : min_index[level_mapper[t.row()] + 1];
      max_index[level_mapper[t.row()] + 1] =
          max_index[level_mapper[t.row()] + 1] < t.row()
              ? t.row()
              : max_index[level_mapper[t.row()] + 1];
      max_index[level_mapper[t.row()] + 1] =
          max_index[level_mapper[t.row()] + 1] < t.col()
              ? t.col()
              : max_index[level_mapper[t.row()] + 1];
    }
  }
  for (FMCA::Index i = 0; i < max_index.size(); ++i) {
    std::cout << int(i) - 1 << "\t (" << min_index[i] << "," << max_index[i]
              << ")\t  -> ";
    const FMCA::Index bsize = max_index[i] - min_index[i] + 1;
    const FMCA::Index oset = min_index[i];
    FMCA::Matrix D(bsize, bsize);
    D.setZero();
    for (const auto &it : matrix_mapper[i]) {
      D(trips[it].row() - oset, trips[it].col() - oset) = trips[it].value();
      D(trips[it].col() - oset, trips[it].row() - oset) = trips[it].value();
    }
    FMCA::Matrix P = D;
    P.setZero();
    P.diagonal().array() = 1. / D.diagonal().array().sqrt();
    // D.diagonal().array() += 1e-14 * NPTS;
    Eigen::JacobiSVD<FMCA::Matrix> svd(D);
    std::cout << "cond: "
              << svd.singularValues()(0) /
                     svd.singularValues()(svd.singularValues().size() - 1)
              << std::endl;
  }
  Eigen::SparseMatrix<FMCA::Scalar> K(NPTS, NPTS);
  K.setFromTriplets(trips.begin(), trips.end());
  FMCA::Matrix Kfull =
      K.selfadjointView<Eigen::Upper>() * FMCA::Matrix::Identity(NPTS, NPTS);
  // Kfull.diagonal().array() += 1e-14 * NPTS;
  // FMCA::Matrix Pr = Kfull;
  // Pr.setZero();
  // Pr.diagonal().array() = 1. / Kfull.diagonal().array().sqrt();

  Eigen::JacobiSVD<FMCA::Matrix> svd(Kfull);
  std::cout << "cond full matrix: "
            << svd.singularValues()(0) /
                   svd.singularValues()(svd.singularValues().size() - 1)
            << std::endl;

  std::cout << std::string(60, '-') << std::endl;
  return 0;
}
