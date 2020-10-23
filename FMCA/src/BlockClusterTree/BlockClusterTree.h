#ifndef FMCA_BLOCKCLUSTERTREE_BLOCKCLUSTERTREE_H_
#define FMCA_BLOCKCLUSTERTREE_BLOCKCLUSTERTREE_H_

namespace FMCA {

/**
 *  \ingroup BlockClusterTree
 *  \brief MatrixBlockCluster is like a triplet to represent
 *         a matrix block which has to be set up
 */
template <typename ClusterTree>
struct MatrixBlockCluster {
  Eigen::Index blockRows;
  Eigen::Index blockCols;
  Eigen::Index startRow;
  Eigen::Index startCol;
  const ClusterTree *rowCluster;
  const ClusterTree *colCluster;
};

/**
 *  \ingroup BlockClusterTree
 *  \brief The BlockClusterTree class manages block cluster trees for
 *         wavelet constructions.
 */
template <typename ClusterTree>
class BlockClusterTree {
 public:
  BlockClusterTree() {}
  BlockClusterTree(const ClusterTree &CT) { init(CT); }
  // top level initialisation method
  void init(const ClusterTree &CT, double a_param = 1.01, double dp_param = 1.5,
            double dt_param = 2, double operator_order = 0) {
    a_param_ = a_param;
    dp_param_ = dp_param;
    dt_param_ = dt_param;
    operator_order_ = operator_order;
    max_wlevel_ = CT.get_max_wlevel();
    cut_const1_ = max_wlevel_ * (dp_param_ - operator_order_) /
                  (dt_param + operator_order_);
    cut_const2_ = (dp_param_ + dt_param_) / 2 * (dt_param + operator_order_);
    computeBlockClusters(CT, CT);
  }
  //////////////////////////////////////////////////////////////////////////////
  // private members
  //////////////////////////////////////////////////////////////////////////////
 private:
  //////////////////////////////////////////////////////////////////////////////
  // admissibility condition
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  // cutoff criterion
  //////////////////////////////////////////////////////////////////////////////
  double cutOffParameter(unsigned int j, unsigned int jp) {
    double first = j < jp ? 1. / (1 << j) : 1. / (1 << jp);
    double second = pow(2., cut_const1_ - (j + jp) * cut_const2_);
    return a_param * (first > second ? first : second);
  }

  bool cutOff(unsigned int j, unsigned int jp, double dist) {
    return cutOffParameter(j, jp) < dist;
  }
  //////////////////////////////////////////////////////////////////////////////
  // private member variables
  //////////////////////////////////////////////////////////////////////////////
  std::vector<std::vector<MatrixBlockCluster<ClusterTree>>> matrix_pattern_;
  double a_param_;
  double dp_param_;
  double dt_param_;
  double operator_order_;
  double max_wlevel_;
  double cut_const1_;
  double cut_const2_;
};
}  // namespace FMCA
#endif
