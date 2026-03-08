// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_MODULUSOFCONTINUITY_EPSILONDISCRETEMODULUSOFCONTINUITY_H_
#define FMCA_MODULUSOFCONTINUITY_EPSILONDISCRETEMODULUSOFCONTINUITY_H_

// #include "../util/Macros.h"

#include "../Clustering/E2LSH.h"

namespace FMCA {

class EpsilonDiscreteModulusOfContinuity {
public:
  EpsilonDiscreteModulusOfContinuity() {};

  template <typename Derived>
  void init(const Matrix &P, const Matrix &f, const Scalar r, const Index R = 2,
            const Scalar TX = 1, const Index min_csize = 1,
            const std::string dx_type = "EUCLIDEAN",
            const std::string dy_type = "EUCLIDEAN",
            const bool add_maxpts = true, const bool use_lsh = false) {
    // set all parameters
    TX_ = TX;
    r_ = r;
    R_ = R;
    K_ = std::ceil(std::log(TX / r) / std::log(R));
    min_csize_ = min_csize;
    tgrid_.resize(K_ + 1);
    omegat_.resize(K_ + 1);
    omegaNk_.resize(K_ + 1);
    XNk_indices_.resize(K_ + 1);
    setDistanceType(dx_, dx_type);
    setDistanceType(dy_, dy_type);

    add_maxpts_ = add_maxpts;
    use_lsh_ = use_lsh;

    if (use_lsh_) {
      lsh_ = E2LSH();
      // using settings as in original  E2LSH paper experimentals
      lsh_.init(P, 10, 30, 4);
    }

    Vector min_dist;
    {
      Derived ct(P, min_csize_);
      min_dist = minDistanceVector(ct, P); // assumes L2 norm (TO BE CHANGED)
    }
    // std::cout << "fill distance: " << min_dist.maxCoeff() << std::endl;
    // std::cout << "separation distance: " << min_dist.minCoeff() << std::endl;
    // std::cout << "number of intervals: " << K_ + 1 << std::endl;
    {
      Derived ct(P, min_csize_);
      // set up modulus of continuity for the full set X = P using the
      // resolution r
      Scalar max_quotient = -1.;
#pragma omp parallel for reduction(max : max_quotient)
      for (Index i = 0; i < P.cols(); ++i) {
        std::vector<Index> nn_idcs;
        if (use_lsh_) {
          nn_idcs = lsh_.computeAENN(P, P.col(i), r_);
        } else {
          nn_idcs = epsNN(ct, P, P.col(i), r_); // assumes L2 norm
        }

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
    }
    // std::cout << X_min_max_[0] << ": (" << P.col(X_min_max_[0]) << ","
    //           << f.col(X_min_max_[0]) << "), " << X_min_max_[1] << ": ("
    //           << P.col(X_min_max_[1]) << "," << f.col(X_min_max_[1]) << ") "
    //           << std::endl;

    for (Index k = 1; k <= K_; ++k) {
      Derived ct(Pprev, min_csize_);
      XNk_indices_[k] = greedySetCovering(ct, Pprev, Rkr);
      // fix indices to become the global indices
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
      Rkr *= R;
      tgrid_[k] = Rkr;
      Matrix Ploc(P.rows(), XNk_indices_[k].size());
      Matrix floc(f.rows(), XNk_indices_[k].size());
      // set up local point set for constructing a cluster tree for epsNN
      for (Index j = 0; j < Ploc.cols(); ++j) {
        Ploc.col(j) = P.col(XNk_indices_[k][j]);
        floc.col(j) = f.col(XNk_indices_[k][j]);
      }
      Pprev = Ploc;
      Derived ctk(Ploc, min_csize_);
      Scalar max_quotient = -1.;
#pragma omp parallel for reduction(max : max_quotient)
      for (Index i = 0; i < Ploc.cols(); ++i) {
        std::vector<Index> nn_idcs;
        if (use_lsh_) {
          nn_idcs = lsh_.computeAENN(Ploc, Ploc.col(i), Rkr);
        } else {
          nn_idcs = epsNN(ctk, Ploc, Ploc.col(i),
                          Rkr); // assumes L2 norm (TO BE CHANGED)
        }

        for (Index j = 0; j < nn_idcs.size(); ++j)
          for (Index k = 0; k < j; ++k) {
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
      // std::cout << "#N" << k << "=" << XNk_indices_[k].size() << std::endl;
    }
  }
  template <typename Derived>
  Scalar omega(Scalar t, const Matrix &P, const Matrix &f) const {
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
    Derived ctk(Ploc, min_csize_);
    Scalar max_quotient = -1.;
#pragma omp parallel for reduction(max : max_quotient)
    for (Index i = 0; i < Ploc.cols(); ++i) {
      std::vector<Index> nn_idcs;
      if (use_lsh_) {
        nn_idcs = lsh_.computeAENN(Ploc, Ploc.col(i), tgrid_[k]);
      } else {
        nn_idcs = epsNN(ctk, Ploc, Ploc.col(i), tgrid_[k]); // assumes L2 norm
      }

      for (Index j = 0; j < nn_idcs.size(); ++j)
        for (Index k = 0; k < j; ++k) {
          const Scalar xdist = dx_(Ploc.col(nn_idcs[j]), Ploc.col(nn_idcs[k]));
          const Scalar fdist = dy_(floc.col(nn_idcs[j]), floc.col(nn_idcs[k]));
          if (xdist <= t)
            max_quotient = max_quotient < fdist ? fdist : max_quotient;
        }
    }
    max_quotient = max_quotient >= 0 ? max_quotient : 0;
    if (k > 0)
      max_quotient =
          max_quotient > omegat_[k - 1] ? max_quotient : omegat_[k - 1];
    return max_quotient;
  }

  const std::vector<std::vector<Index>> &XNk_indices() const {
    return XNk_indices_;
  }

  const std::vector<Scalar> &omegaNk() const { return omegaNk_; }

  const std::vector<Scalar> &omegat() const { return omegat_; }
  const std::vector<Scalar> &tgrid() const { return tgrid_; }

private:
  void
  setDistanceType(std::function<Scalar(const Vector &, const Vector &)> &df,
                  const std::string &dist_type) {
    if (dist_type == "EUCLIDEAN") {
      df = [](const Vector &x, const Vector &y) { return (x - y).norm(); };
    } else if (dist_type == "GEODESIC") {
      df = [](const Vector &x, const Vector &y) {
        return SphereClusterTree::geodesicDistance(x, y);
      };
    } else
      assert(false && "desired distance not implemented");
    return;
  }

  std::vector<std::vector<Index>> XNk_indices_;
  std::vector<Index> X_min_max_;
  std::vector<Scalar> omegaNk_;
  std::vector<Scalar> tgrid_;
  std::vector<Scalar> omegat_;
  std::function<Scalar(const Vector &, const Vector &)> dx_;
  std::function<Scalar(const Vector &, const Vector &)> dy_;
  Scalar TX_;
  Scalar r_;
  Index R_;
  Index K_;
  Index min_csize_;
  bool add_maxpts_;
  bool use_lsh_;
  E2LSH lsh_;
};
} // namespace FMCA
#endif
