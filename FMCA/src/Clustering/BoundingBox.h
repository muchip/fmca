#ifndef FMCA_CLUSTERTREE_BOUNDINGBOX_H_
#define FMCA_CLUSTERTREE_BOUNDINGBOX_H_

namespace FMCA {
/**
 *  \ingroup ClusterTree
 *  \brief The BoundingBox class manages bounding boxes in arbitrary dimension.
 *         The only subdivision operation implemented so far is based on
 *         geometric bisection.
 */
template <typename T, unsigned int Dim>
class BoundingBox {
 public:
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  BoundingBox(){};

  BoundingBox(const Eigen::Matrix<T, Dim, Eigen::Dynamic>& P) { init(P); }

  BoundingBox(const BoundingBox<T, Dim>& other) { bb_ = other.bb_; }

  BoundingBox(BoundingBox<T, Dim>&& other) { bb_ = std::move(other.bb_); }

  BoundingBox<T, Dim>& operator=(BoundingBox<T, Dim> other) {
    bb_.swap(other.bb_);
    return *this;
  }
  //////////////////////////////////////////////////////////////////////////////
  // init method
  //////////////////////////////////////////////////////////////////////////////
  void init(const Eigen::Matrix<T, Dim, Eigen::Dynamic>& P) {
    for (auto i = 0; i < Dim; ++i) {
      bb_(i, 0) = P.row(i).minCoeff();
      bb_(i, 1) = P.row(i).maxCoeff();
      // add some padding, e.g. 5%
      bb_(i, 0) =
          bb_(i, 0) < 0 ? bb_(i, 0) * (1 + 5e-2) : bb_(i, 0) * (1 - 5e-2);
      bb_(i, 1) =
          bb_(i, 1) < 0 ? bb_(i, 1) * (1 - 5e-2) : bb_(i, 1) * (1 + 5e-2);
    }
    bb_.col(2) = bb_.col(1) - bb_.col(0);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // split (based on geometric bisection)
  //////////////////////////////////////////////////////////////////////////////
  BoundingBox<T, Dim> split(bool index) const {
    BoundingBox<T, Dim> retval(*this);
    unsigned int longest;
    bb_.col(2).maxCoeff(&longest);
    if (index == false) {
      retval.bb_(longest, 2) *= 0.5;
      retval.bb_(longest, 1) -= retval.bb_(longest, 2);
      return retval;
    } else {
      retval.bb_(longest, 2) *= 0.5;
      retval.bb_(longest, 0) += retval.bb_(longest, 2);
      return retval;
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  // setter and getter
  //////////////////////////////////////////////////////////////////////////////
  void set_bb(const Eigen::Matrix<T, Dim, 3u>& bbmat) {
    bb_ = bbmat;
    return;
  }

  const Eigen::Matrix<T, Dim, 3u>& get_bb() const { return bb_; }

  Eigen::Matrix<T, Dim, 3u>& get_bb() { return bb_; }
  //////////////////////////////////////////////////////////////////////////////
  /// private members
  //////////////////////////////////////////////////////////////////////////////
 private:
  // format is [lower, upper, dist]
  Eigen::Matrix<T, Dim, 3u> bb_;
};
}  // namespace FCMA
#endif
