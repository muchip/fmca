// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 206, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSORBASE_H_
#define FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSORBASE_H_

namespace FMCA {

template <typename Derived>
class SampletMatrixCompressorBase {
 public:
  Derived &derived() { return *static_cast<Derived *>(this); }
  // return a const reference to the derived object
  const Derived &derived() const { return *static_cast<const Derived *>(this); }
  //////////////////////////////////////////////////////////////////////////////
  // exposed the trees init routine
  template <typename... Ts>
  void init(Ts &&...ts) {
    derived().init(std::forward<Ts>(ts)...);
  }

  const std::vector<Triplet> &triplets() const { return triplet_list_; }

  std::vector<Triplet> release_triplets() {
    std::vector<Triplet> retval;
    std::swap(triplet_list_, retval);
    return retval;
  }

  Index rows() const { return mpts_; }
  Index cols() const { return npts_; }
  Scalar threshold() const { return threshold_; }
  Scalar eta() const { return eta_; }

  std::vector<Triplet> aposteriori_triplets_fast(const Scalar thres) {
    std::vector<Triplet> retval;
    std::vector<std::vector<Index>> buckets(17);
    std::vector<Scalar> norms2(17);
    const Scalar invlog10 = 1. / std::log(10.);
    for (FMCA::Index i = 0; i < triplet_list_.size(); ++i) {
      const Scalar entry = std::abs(triplet_list_[i].value());
      const Scalar val = std::min(-std::floor(invlog10 * std::log(entry)), 16.);
      const Index ind = val < 0 ? 0 : val;
      buckets[ind].push_back(i);
      norms2[ind] += entry * entry;
    }
    Scalar fnorm2 = 0;
    for (int i = 16; i >= 0; --i) fnorm2 += norms2[i];
    Scalar cut_snorm = 0;
    Index cut_off = 17;
    for (int i = 16; i >= 0; --i) {
      cut_snorm += norms2[i];
      if (std::sqrt(cut_snorm / fnorm2) >= thres) break;
      --cut_off;
    }
    Index ntriplets = 0;
    for (Index i = 0; i < cut_off; ++i) ntriplets += buckets[i].size();
    retval.reserve(ntriplets + npts_);
    for (Index i = 0; i < cut_off; ++i)
      for (const auto &it : buckets[i]) retval.push_back(triplet_list_[it]);
    // make sure the matrix contains the diagonal
    for (Index i = cut_off; i < 17; ++i)
      for (const auto &it : buckets[i])
        if (triplet_list_[it].row() == triplet_list_[it].col())
          retval.push_back(triplet_list_[it]);
    retval.shrink_to_fit();
    return retval;
  }

  std::vector<Triplet> aposteriori_triplets(const Scalar thres) {
    std::vector<Triplet> triplets = triplet_list_;
    if (std::abs(thres) < FMCA_ZERO_TOLERANCE) return triplets;

    // sort the triplets by magnitude, putting diagonal entries first
    // note that first sorting and then summing small to large makes
    // everything stable (positive numbers). Using Kahan summation did
    // not further improve afterwards, so we stay with fast summation
    std::vector<long int> idcs(triplet_list_.size());
    std::iota(idcs.begin(), idcs.end(), 0);
    {
      struct comp {
        comp(const std::vector<Triplet> &triplets) : ts_(triplets) {}
        bool operator()(const Index &a, const Index &b) const {
          const Scalar val1 = (ts_[a].row() == ts_[a].col())
                                  ? FMCA_INF
                                  : std::abs(ts_[a].value());
          const Scalar val2 = (ts_[b].row() == ts_[b].col())
                                  ? FMCA_INF
                                  : std::abs(ts_[b].value());
          return val1 > val2;
        }
        const std::vector<Triplet> &ts_;
      };
      std::sort(idcs.begin(), idcs.end(), comp(triplet_list_));
    }

    Scalar squared_norm = 0;
    for (auto it = idcs.rbegin(); it != idcs.rend(); ++it)
      squared_norm += triplet_list_[*it].value() * triplet_list_[*it].value();

    Scalar cut_snorm = 0;
    Index cut_off = triplet_list_.size();
    for (auto it = idcs.rbegin(); it != idcs.rend(); ++it) {
      cut_snorm += triplet_list_[*it].value() * triplet_list_[*it].value();
      if (std::sqrt(cut_snorm / squared_norm) >= thres) break;
      --cut_off;
    }
    // keep at least the diagonal
    cut_off = cut_off < npts_ ? npts_ : cut_off;
    idcs.resize(cut_off);
    triplets.resize(cut_off);
    for (Index i = 0; i < cut_off; ++i) triplets[i] = triplet_list_[idcs[i]];
    return triplets;
  }

 protected:
  void clearTriplets() { triplet_list_.clear(); }

  void reserveTriplets(std::size_t n) { triplet_list_.reserve(n); }

  void appendTriplets(std::vector<Triplet> &&ts) {
    triplet_list_.insert(triplet_list_.end(),
                         std::make_move_iterator(ts.begin()),
                         std::make_move_iterator(ts.end()));
  }
  void pushTriplet(Index r, Index c, Scalar v) {
    triplet_list_.emplace_back(r, c, v);
  }
  void setThreshold(Index m, Index n, Scalar threshold, Scalar eta) {
    mpts_ = m;
    npts_ = n;
    threshold_ = threshold;
    eta_ = eta;
    return;
  }

  void setDimensions(Index m, Index n) {
    mpts_ = m;
    npts_ = n;
    return;
  }
  void setThreshold(Scalar threshold) {
    threshold_ = threshold;
    return;
  }

  void setEta(Scalar eta) {
    eta_ = eta;
    return;
  }

  /**
   *  \brief writes a given matrix block into a-posteriori thresholded
   *         triplet format
   **/
  template <typename otherDerived>
  void storeBlock(std::vector<Triplet> &triplet_buffer, Index srow, Index scol,
                  Index nrows, Index ncols,
                  const MatrixBase<otherDerived> &block) {
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if ((std::abs(block(j, k)) > threshold_) || (srow == scol && j == k))
          triplet_buffer.push_back(Triplet(srow + j, scol + k, block(j, k)));
  }
  /**
   *  \brief writes a given matrix block into a-posteriori thresholded
   *         triplet format
   **/
  template <typename otherDerived>
  void storeSymBlock(std::vector<Triplet> &triplet_buffer, Index srow,
                     Index scol, Index nrows, Index ncols,
                     const MatrixBase<otherDerived> &block) {
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if ((srow + j <= scol + k && std::abs(block(j, k)) > threshold_) ||
            (srow == scol && j == k))
          triplet_buffer.push_back(Triplet(srow + j, scol + k, block(j, k)));
  }

  void storeEmptyBlock(std::vector<Triplet> &triplet_buffer, Index srow,
                       Index scol, Index nrows, Index ncols) {
    for (Index k = 0; k < ncols; ++k)
      for (Index j = 0; j < nrows; ++j)
        triplet_buffer.push_back(Triplet(srow + j, scol + k, 0));
  }

  void storeSymEmptyBlock(std::vector<Triplet> &triplet_buffer, Index srow,
                          Index scol, Index nrows, Index ncols) {
    for (Index k = 0; k < ncols; ++k)
      for (Index j = 0; j < nrows; ++j)
        if (srow + j <= scol + k)
          triplet_buffer.push_back(Triplet(srow + j, scol + k, 0));
  }

 private:
  std::vector<Triplet> triplet_list_;
  Scalar eta_;
  Scalar threshold_;
  Index mpts_;
  Index npts_;
};

}  // namespace FMCA
#endif
