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
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 10000

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const std::string kernels[2] = {"TPS2D", "BIHARMONIC3D"};
  const FMCA::Index dims[2] = {2, 3};
  const FMCA::Index dtilde = 4;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const FMCA::Scalar eta = 0.5;
  const FMCA::Scalar threshold = 1e-8;

  for (FMCA::Index t = 0; t < 2; ++t) {
    const FMCA::Index dim = dims[t];
    const FMCA::Index mq = 1 + dim;  // monomials {1, x_1, ..., x_dim}
    const FMCA::CovarianceKernel function(kernels[t], 1.);
    const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(dim, NPTS).array() + 1);
    std::cout << "kernel:                       " << kernels[t] << std::endl;
    std::cout << "dimension:                    " << dim << std::endl;
    std::cout << "dtilde:                       " << dtilde << std::endl;
    std::cout << "eta:                          " << eta << std::endl;
    const Moments mom(P, mpole_deg);
    const MatrixEvaluator mat_eval(mom, function);
    const SampletMoments samp_mom(P, dtilde - 1);
    H2SampletTree hst(mom, samp_mom, 0, P);

    // decoupling of the polynomial block:  T P = [R; 0]
    FMCA::Matrix Pol(mq, NPTS);
    for (FMCA::Index i = 0; i < NPTS; ++i) {
      Pol(0, i) = 1.0;
      Pol.block(1, i, dim, 1) = P.col(i);
    }
    const FMCA::Matrix TPol =
        hst.sampletTransform(hst.toClusterOrder(Pol.transpose()));
    const FMCA::Matrix R = TPol.topRows(mq);
    const FMCA::Scalar dec_err = TPol.bottomRows(NPTS - mq).norm() / TPol.norm();
    std::cout << "decoupling error TP=[R;0]:    " << dec_err << std::endl;
    assert(dec_err < 1e-10 && "polynomial block not annihilated");

    // samplet compression of the CPD kernel matrix
    T.tic();
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> Scomp;
    Scomp.init(hst, eta, 100 * FMCA_ZERO_TOLERANCE);
    Scomp.compress(mat_eval);
    Scomp.triplets();
    const auto &trips = Scomp.aposteriori_triplets_fast(threshold);
    T.toc("compression:                 ");
    std::cout << "anz (a-posteriori):           "
              << std::round(trips.size() / FMCA::Scalar(NPTS)) << std::endl;
    Eigen::SparseMatrix<FMCA::Scalar> Smat(NPTS, NPTS);
    Smat.setFromTriplets(trips.begin(), trips.end());
    {
      FMCA::Vector x(NPTS), y1(NPTS), y2(NPTS);
      FMCA::Scalar err = 0, nrm = 0;
      for (FMCA::Index i = 0; i < 10; ++i) {
        const FMCA::Index index = rand() % NPTS;
        x.setZero();
        x(index) = 1;
        const FMCA::Vector col = function.eval(P, P.col(hst.indices()[index]));
        y1 = col(
            Eigen::Map<const FMCA::iVector>(hst.indices(), hst.block_size()));
        x = hst.sampletTransform(x);
        y2.setZero();
        for (const auto &it : trips) {
          y2(it.row()) += it.value() * x(it.col());
          if (it.row() != it.col()) y2(it.col()) += it.value() * x(it.row());
        }
        y2 = hst.inverseSampletTransform(y2);
        err += (y1 - y2).squaredNorm();
        nrm += y1.squaredNorm();
      }
      err = sqrt(err / nrm);
      std::cout << "compression error:            " << err << std::endl;
      assert(err < 1e-4 && "compression error too large");
    }

    // positive definiteness of the detail block K_PsiPsi
    Eigen::SparseMatrix<FMCA::Scalar> Kpsi =
        Smat.block(mq, mq, NPTS - mq, NPTS - mq);
    const Eigen::SparseMatrix<FMCA::Scalar> K_PPsi =
        Smat.block(0, mq, mq, NPTS - mq);
    // add regularization to the detail block
    {
      Eigen::SparseMatrix<FMCA::Scalar> I(NPTS - mq, NPTS - mq);
      I.setIdentity();
      Kpsi += I * 1e-4;
    }
    Eigen::SimplicialLLT<Eigen::SparseMatrix<FMCA::Scalar>, Eigen::Upper> llt;
    llt.compute(Kpsi);
    std::cout << "Cholesky of K_PsiPsi:         "
              << (llt.info() == Eigen::Success ? "SUCCESS" : "FAILED")
              << std::endl;
    assert(llt.info() == Eigen::Success && "Cholesky failed");

    // null space solve of the saddle point system, c_P = 0
    FMCA::Vector y(NPTS);
    for (FMCA::Index i = 0; i < NPTS; ++i)
      y(i) = std::sin(3 * P(0, i)) * std::exp(-P(dim - 1, i));
    T.tic();
    const FMCA::Vector Uy = hst.sampletTransform(hst.toClusterOrder(y));
    const FMCA::Vector c_Psi = llt.solve(Uy.tail(NPTS - mq));
    FMCA::Vector c_samplet = FMCA::Vector::Zero(NPTS);
    c_samplet.tail(NPTS - mq) = c_Psi;
    const FMCA::Vector c =
        hst.toNaturalOrder(hst.inverseSampletTransform(c_samplet));
    const FMCA::Vector d =
        FMCA::Matrix(R).fullPivLu().solve(Uy.head(mq) - K_PPsi * c_Psi);
    T.toc("null space solve:            ");
    const FMCA::Scalar side_err = (Pol * c).norm() / c.norm();
    std::cout << "side condition |P'c|:         " << side_err << std::endl;

    // polynomial reproduction: for y in P_1 the kernel part has to vanish
    {
      FMCA::Vector beta(mq);
      for (FMCA::Index i = 0; i < mq; ++i) beta(i) = 1.0 + i;
      const FMCA::Vector yp = Pol.transpose() * beta;
      const FMCA::Vector Uyp = hst.sampletTransform(hst.toClusterOrder(yp));
      const FMCA::Vector cp_Psi = llt.solve(Uyp.tail(NPTS - mq));
      const FMCA::Vector dp =
          FMCA::Matrix(R).fullPivLu().solve(Uyp.head(mq) - K_PPsi * cp_Psi);
      std::cout << "kernel part |c| for y in P_1: " << cp_Psi.norm() / yp.norm()
                << std::endl;
      std::cout << "drift error |d - beta|:       "
                << (dp - beta).norm() / beta.norm() << std::endl;
      assert(cp_Psi.norm() / yp.norm() < 1e-10 && "kernel part not vanishing");
      assert((dp - beta).norm() / beta.norm() < 1e-10 && "drift not recovered");
    }
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}
