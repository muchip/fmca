#ifndef BEMBEL_HMATRIX_TREELEAF_H_
#define BEMBEL_HMATRIX_TREELEAF_H_

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>

template <typename Derived>
struct DummyTruncator {
  static void lowRankTruncation(Derived &L, Derived &R) { return; }
  static void fullTruncation(Derived &L, Derived &R, Derived &F) {
    L = F;
    R = Derived::Identity(F.cols(), F.cols());
    return;
  }
};
/**   \brief class that contains the treeleafs consists of
 *           a full matrix and a low-rank matrix the flag
 *           _lowRank indicates which one of them is used
 */
template <typename Derived, typename Truncator>
class TreeLeaf {
 public:
  //////////////////////////////////////////////////////////////////////////////
  //    Constructors
  //////////////////////////////////////////////////////////////////////////////
  /**
   * \brief void constructor
   **/
  TreeLeaf(void)
      : F_(Derived(0, 0)),
        L_(Derived(0, 0)),
        R_(Derived(0, 0)),
        isLowRank_(false) {}
  /**
   * \brief copy constructor
   **/

  TreeLeaf(TreeLeaf &other)
      : F_(other.F_),
        L_(other.L_),
        R_(other.R_),
        isLowRank_(other.isLowRank_) {}
  /**
   * \brief move constructor
   **/
  TreeLeaf(TreeLeaf &&other)
      : F_(std::move(other.F_)),
        L_(std::move(other.L_)),
        R_(std::move(other.R_)),
        isLowRank_(other.isLowRank_) {}
  /**
   * \brief lowRank constructor
   *        whatever Eigen object is put in here will be evaluated
   */
  template <typename otherDerived>
  TreeLeaf(const Eigen::MatrixBase<otherDerived> &L,
           const Eigen::MatrixBase<otherDerived> &R)
      : F_(Derived(0, 0)), L_(L), R_(R), isLowRank_(true) {}
  /**
   * \brief full constructor
   *        whatever Eigen object is put in here will be evaluated
   **/
  template <typename otherDerived>
  TreeLeaf(const Eigen::MatrixBase<otherDerived> &F)
      : F_(F), L_(Derived(0, 0)), R_(Derived(0, 0)), isLowRank_(false) {}
  /**
   * \brief lowRank move constructor
   **/
  TreeLeaf(Derived &&L, Derived &&R)
      : F_(Derived(0, 0)), L_(L), R_(R), isLowRank_(true) {}
  /**
   * \brief full move constructor
   **/
  TreeLeaf(Derived &&F)
      : F_(F), L_(Derived(0, 0)), R_(Derived(0, 0)), isLowRank_(false) {}
  //////////////////////////////////////////////////////////////////////////////
  //    Operators
  //////////////////////////////////////////////////////////////////////////////
  /**
   * \brief assignment operator, works for copy and move assignment
   **/
  TreeLeaf &operator=(TreeLeaf other) {
    F_.swap(other.F_);
    L_.swap(other.L_);
    R_.swap(other.R_);
    std::swap(isLowRank_, other.isLowRank_);
    return *this;
  }
  /**   \brief += assignment operator
   */
  TreeLeaf &operator+=(const TreeLeaf &other) {
    if (isLowRank_) {
      if (other.isLowRank_) {
        auto LcolsOld = L_.cols();
        auto RcolsOld = R_.cols();
        L_.conservativeResize(L_.rows(), L_.cols() + other.L_.cols());
        R_.conservativeResize(R_.rows(), R_.cols() + other.R_.cols());
        L_.block(0, LcolsOld, L_.rows(), other.L_.cols()) = other.L_;
        R_.block(0, RcolsOld, R_.rows(), other.R_.cols()) = other.R_;
        Truncator::lowRankTruncation(L_, R_);
      } else {
        Eigen::MatrixXd tmp = other.F_ + L_ * R_.transpose();
        Truncator::fullTruncation(L_, R_, tmp);
      }
    } else {
      if (other.isLowRank_)
        F_ += other.L_ * other.R_.transpose();
      else
        F_ += other.F_;
    }
    return *this;
  }
  /**   \brief addition operator
   */
  TreeLeaf operator+(const TreeLeaf &other) {
    TreeLeaf retVal(other);
    return retVal += *this;
  }
  template <typename otherDerived>
  Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
  operator*(const Eigen::MatrixBase<otherDerived> &M) {
    if (isLowRank_)
      return L_ * (R_.transpose() * M);
    else
      return (F_ * M);
  }

 private:
  Derived F_, L_, R_;
  bool isLowRank_;
};

#endif
