// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer, Michele Palma
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_MODULUSOFCONTINUITY_LSHDISCRETEMODULUSOFCONTINUITY_H_
#define FMCA_MODULUSOFCONTINUITY_LSHDISCRETEMODULUSOFCONTINUITY_H_

#include "../Clustering/E2LSH.h"
#include "../Clustering/greedySetCovering.h"
#include "../util/Macros.h"
#include "DiscreteModulusOfContinuityBase.h"

namespace FMCA {

class LSHDiscreteModulusOfContinuity
    : public DiscreteModulusOfContinuityBase<LSHDiscreteModulusOfContinuity> {
public:
  using Base = DiscreteModulusOfContinuityBase<LSHDiscreteModulusOfContinuity>;

  LSHDiscreteModulusOfContinuity() {}

  void init(const Matrix &P, const Matrix &f, const Scalar TX, const Scalar r,
            const Index R = 2, const Index min_csize = 1,
            const bool add_maxpts = true, const Index kl = 10,
            const Index L = 30, const Scalar w = 4) {
    setDistanceType(dx_, "EUCLIDEAN");
    setDistanceType(dy_, "EUCLIDEAN");

    TX_ = TX > 0 ? TX : 0;
    bb_.resize(P.rows(), 3);
    bb_.col(0) = P.rowwise().minCoeff();
    bb_.col(1) = P.rowwise().maxCoeff();
    bb_.col(2) = bb_.col(1) - bb_.col(0);
    const Scalar bb_diam = bb_.col(2).norm();
    TX_ = TX_ > bb_diam ? bb_diam : TX_;
    if (TX_ <= 0) {
      Base::tgrid_.resize(1, 0);
      Base::omegat_.resize(1, 0);
      return;
    }

    // step_size_ = step_size <= TX_ ? step_size : TX_; //must be defined as
    // here it changes....

    // set all parameters
    r_ = r; // also initial step_size
    R_ = R;
    min_csize_ = min_csize;
    add_maxpts_ = add_maxpts;

    // const Index nbins = ... + 1;
    K_ = std::ceil(std::log(TX_ / r_) / std::log(R_));
    Base::tgrid_.resize(K_ + 1);
    Base::omegat_.resize(K_ + 1);
    omegaNk_.resize(K_ + 1);
    XNk_indices_.resize(K_ + 1);

    lsh_ = E2LSH();
    // using settings as in original  E2LSH paper experimentals init(P,10,30,4)
    lsh_.init(P, kl, L, w);

    // it shall all happen in init procedure
    {

      // set up modulus of continuity for the full set X = P using the
      // resolution r
      Scalar max_quotient = -1.;
#pragma omp parallel for reduction(max : max_quotient)
      for (Index i = 0; i < P.cols(); ++i) {
        std::vector<Index> nn_idcs;

        nn_idcs = lsh_.computeAENN(P, P.col(i), r_); // assumes L2 norm

        for (Index j = 0; j < nn_idcs.size(); ++j)
          for (Index k = 0; k < j; ++k) {
            const Scalar xdist = dx_(P.col(nn_idcs[j]), P.col(nn_idcs[k]));
            assert(xdist <= 2 * r_ && "error");
            const Scalar fdist = dy_(f.col(nn_idcs[j]), f.col(nn_idcs[k]));
            if (xdist <= r_)
              max_quotient = max_quotient < fdist ? fdist : max_quotient;
          }
      }
      omegaNk_[0] = max_quotient;
      tgrid_[0] = r;
      omegat_[0] = omegaNk_[0];
      XNk_indices_[0].resize(P.cols());
      std::iota(XNk_indices_[0].begin(), XNk_indices_[0].end(), 0);
    }

    // std::cout << "#N0=" << XNk_indices_[0].size() << std::endl;
    //  compute reduced index sets using greedy method and compute corresponding
    //  moduli of continuity
    Scalar Rkr = r_;
    Matrix Pprev = P;
    X_min_max_.resize(2);
    FMCA::Scalar max = -FMCA_INF;
    FMCA::Scalar min = FMCA_INF;
    for (FMCA::Index i = 0; i < f.cols(); ++i) {
      Scalar normfi =
          dy_(f.col(i),
              Vector::Zero(f.rows())); // norm using dy_ distance function
                                       // (assuming it is possible to do so)

      if (normfi < min) {
        min = normfi;
        X_min_max_[0] = i;
      }
      if (normfi > max) {
        max = normfi;
        X_min_max_[1] = i;
      }
      //
    }

    // block2
    for (Index k = 1; k <= K_; ++k) {

      // auto indices_global = greedySetCoveringLSH(lsh_, Pprev, P, Rkr);
      //  map back the indices from P to those valid in Pprev
      //  same as mapping indices to X_Nk (just exclude those points that are
      //  not there) fix indices to become the global indices std::vector<Index>
      //  lsh_nn_idcs = indices_global; std::vector<Index> nn_idcs;
      //  std::vector<Index> nn_idcs_global;
      //  std::set<Index> ploc_set(XNk_indices_[k].begin(),
      //  XNk_indices_[k].end()); std::set<Index> lsh_set(lsh_nn_idcs.begin(),
      //  lsh_nn_idcs.end());

      // std::set_intersection(ploc_set.begin(), ploc_set.end(),
      // lsh_set.begin(),
      //                       lsh_set.end(),
      //                       std::back_inserter(nn_idcs_global));

      // nn_idcs.clear();

      // for (auto global_idx : nn_idcs_global) {
      //   auto it = std::lower_bound(XNk_indices_[k].begin(),
      //                              XNk_indices_[k].end(), global_idx);
      //   if (it != XNk_indices_[k].end() && *it == global_idx) {
      //     nn_idcs.push_back(std::distance(XNk_indices_[k].begin(), it));
      //   }
      // }

      // store local indices (end)
      // XNk_indices_[k] = indices_local;
      // first take intersection for compatible indices
      XNk_indices_[k].resize(P.cols()); // fake line to be removed
      std::iota(XNk_indices_[k].begin(), XNk_indices_[k].end(),
                0); // fake line to be removed

      bool hasmin = false;
      bool hasmax = false;
      for (Index j = 0; j < XNk_indices_[k].size(); ++j) {

        XNk_indices_[k][j] = XNk_indices_[k - 1][XNk_indices_[k][j]];

        if (XNk_indices_[k][j] == X_min_max_[0])
          hasmin = true;
        if (XNk_indices_[k][j] == X_min_max_[1])
          hasmax = true;
      }
      if (add_maxpts_) {
        if (!hasmin)
          XNk_indices_[k].push_back(X_min_max_[0]);
        if (!hasmax)
          XNk_indices_[k].push_back(X_min_max_[1]);
      }

      std::sort(XNk_indices_[k].begin(), XNk_indices_[k].end());

      // std::cout << "for" << k << " we have " << XNk_indices_[k].size()
      //           << " pointsss";

      Rkr *= R;
      tgrid_[k] = Rkr;
      Matrix Ploc(
          P.rows(),
          XNk_indices_[k].size()); // e.g. just points from XNk_indices_[k]
      Matrix floc(f.rows(), XNk_indices_[k].size());
      // set up local point set for constructing a cluster tree for epsNN
      for (Index j = 0; j < Ploc.cols(); ++j) {
        Ploc.col(j) = P.col(XNk_indices_[k][j]);
        floc.col(j) = f.col(XNk_indices_[k][j]);
      }

      // std::cout << "for" << k << " the construction of Ploc, floc, was okay";

      Pprev = Ploc;
      // DerivedCT ctk(Ploc, min_csize_);

      //  COMMENT BLOCK A (computation of moc on the reduced set)
      Scalar max_quotient = -1.;
#pragma omp parallel for reduction(max : max_quotient)
      for (Index i = 0; i < Ploc.cols(); ++i) {
        std::vector<Index> nn_idcs;
        std::vector<Index> lsh_nn_idcs;

        lsh_nn_idcs = lsh_.computeAENN(P, Ploc.col(i), Rkr);
        std::vector<Index> nn_idcs_global;
        // we can probably do this via binary search, given ordered vectors
        std::set<Index> ploc_set(XNk_indices_[k].begin(),
                                 XNk_indices_[k].end());
        std::set<Index> lsh_set(lsh_nn_idcs.begin(), lsh_nn_idcs.end());

        std::set_intersection(ploc_set.begin(), ploc_set.end(), lsh_set.begin(),
                              lsh_set.end(),
                              std::back_inserter(nn_idcs_global));

        nn_idcs.clear();

        for (auto global_idx : nn_idcs_global) {
          auto it = std::lower_bound(XNk_indices_[k].begin(),
                                     XNk_indices_[k].end(), global_idx);
          if (it != XNk_indices_[k].end() && *it == global_idx) {
            nn_idcs.push_back(std::distance(XNk_indices_[k].begin(), it));
          }
        }

        // nn_idcs = epsNN(ctk, Ploc, Ploc.col(i),
        //                 Rkr); // assumes L2 norm

        for (Index j = 0; j < nn_idcs.size(); ++j)
          for (Index k = 0; k < j; ++k) {

            // std::cout << " nn_idcs[j] = " << nn_idcs[j] << ";";

            const Scalar xdist =
                dx_(Ploc.col(nn_idcs[j]), Ploc.col(nn_idcs[k]));
            assert(xdist <= 2 * Rkr && "error");
            const Scalar fdist =
                dy_(floc.col(nn_idcs[j]), floc.col(nn_idcs[k]));
            if (xdist <= Rkr)
              max_quotient = max_quotient < fdist ? fdist : max_quotient;
          }
      }
      omegaNk_[k] = max_quotient;
      omegat_[k] = omegaNk_[k] > omegat_[k - 1] ? omegaNk_[k] : omegat_[k - 1];
      // DECOMMENT BLOCK A
      // std::cout << "#N" << k << "=" << XNk_indices_[k].size() << std::endl;
    }
    // std::cout << "init finished!";
  }

  Scalar omega(Scalar t, const Matrix &P, const Matrix &f) const {
    // std::cout << "omega started! for " << t << " ";

    t = (t >= 0 ? t : 0);
    Scalar retval = 0;
    Index k = 0;
    // find interval
    if (t > TX_)
      k = K_;
    else
      while (t > tgrid_[k])
        ++k;
    Matrix Ploc(P.rows(), XNk_indices_[k].size());
    Matrix floc(f.rows(), XNk_indices_[k].size());
    // set up local point set for constructing a cluster tree for epsNN

    for (Index j = 0; j < Ploc.cols(); ++j) {
      Ploc.col(j) = P.col(XNk_indices_[k][j]);
      floc.col(j) = f.col(XNk_indices_[k][j]);
    }

    Scalar max_quotient = -1.;
#pragma omp parallel for reduction(max : max_quotient)
    for (Index i = 0; i < Ploc.cols(); ++i) {

      std::vector<Index> lsh_nn_idcs =
          lsh_.computeAENN(P, Ploc.col(i), tgrid_[k]); // shoulnd't be t

      std::vector<Index> nn_idcs;
      std::vector<Index> nn_idcs_global;
      std::set<Index> ploc_set(XNk_indices_[k].begin(), XNk_indices_[k].end());
      std::set<Index> lsh_set(lsh_nn_idcs.begin(), lsh_nn_idcs.end());

      std::set_intersection(ploc_set.begin(), ploc_set.end(), lsh_set.begin(),
                            lsh_set.end(), std::back_inserter(nn_idcs_global));

      nn_idcs.clear();

      for (auto global_idx : nn_idcs_global) {
        auto it = std::lower_bound(XNk_indices_[k].begin(),
                                   XNk_indices_[k].end(), global_idx);
        if (it != XNk_indices_[k].end() && *it == global_idx) {
          nn_idcs.push_back(std::distance(XNk_indices_[k].begin(), it));
        }
      }

      for (Index j = 0; j < nn_idcs.size(); ++j)
        for (Index k = 0; k < j; ++k) {
          const Scalar xdist = dx_(Ploc.col(nn_idcs[j]), Ploc.col(nn_idcs[k]));
          const Scalar fdist = dy_(floc.col(nn_idcs[j]), floc.col(nn_idcs[k]));
          if (xdist <= t)
            max_quotient = max_quotient < fdist ? fdist : max_quotient;
        }
    }
    // std::cout << "good omega";
    max_quotient = max_quotient >= 0 ? max_quotient : 0;
    if (k > 0)
      max_quotient =
          max_quotient > omegat_[k - 1] ? max_quotient : omegat_[k - 1];
    return max_quotient;
  }

private:
  using Base::bb_;
  using Base::dx_;
  using Base::dy_;
  using Base::omegat_;
  using Base::setDistanceType;
  using Base::step_size_;
  using Base::tgrid_;
  using Base::TX_;

  std::vector<std::vector<Index>> XNk_indices_;
  std::vector<Index> X_min_max_;
  std::vector<Scalar> omegaNk_;
  Scalar r_;
  Index R_;
  Index K_;
  Index min_csize_;
  bool add_maxpts_;
  E2LSH lsh_;
};

} // namespace FMCA
#endif
