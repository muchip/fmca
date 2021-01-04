#ifndef FMCA_BLOCKCLUSTERTREE_BLOCKCLUSTERTREE_H_
#define FMCA_BLOCKCLUSTERTREE_BLOCKCLUSTERTREE_H_

namespace FMCA {

/**
 *  \ingroup BlockClusterTree
 *  \brief MatrixBlockCluster is like a triplet to represent
 *         a matrix block which has to be set up
 */
template <typename ClusterTree> struct MatrixBlockCluster {
  Eigen::Index blockRows;
  Eigen::Index blockCols;
  Eigen::Index startRow;
  Eigen::Index startCol;
  const ClusterTree *rowCluster;
  const ClusterTree *colCluster;
};
/**
 *  \ingroup BlockClusterTree
 *  \brief BlockClusterTreeData is a struct which holds the underlying data
 *         for the construction of the block cluster tree
 */
struct BlockClusterTreeData {
  //////////////////////////////////////////////////////////////////////////////
  /// constructor
  //////////////////////////////////////////////////////////////////////////////
  BlockClusterTreeData(double a_param, double dp_param, double dt_param,
                       double operator_order, unsigned int max_wlevel)
      : a_param_(a_param), dp_param_(dp_param), dt_param_(dt_param),
        operator_order_(operator_order), max_wlevel_(max_wlevel),
        cut_const1_(max_wlevel * (dp_param - operator_order) /
                    (dt_param + operator_order)),
        cut_const2_((dp_param + dt_param) / 2 * (dt_param + operator_order)) {}
  //////////////////////////////////////////////////////////////////////////////
  /// cutOff criterion
  //////////////////////////////////////////////////////////////////////////////
  double cutOffParameter(unsigned int j, unsigned int jp) {
    const double first = j < jp ? 1. / (1 << j) : 1. / (1 << jp);
    const double second = std::pow(2., cut_const1_ - (j + jp) * cut_const2_);
    return a_param_ * (first > second ? first : second);
  }
  //////////////////////////////////////////////////////////////////////////////
  bool cutOff(unsigned int j, unsigned int jp, double dist) {
    return cutOffParameter(j, jp) < dist;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  const double a_param_;
  const double dp_param_;
  const double dt_param_;
  const double operator_order_;
  const unsigned int max_wlevel_;
  const double cut_const1_;
  const double cut_const2_;
};

/**
 *  \ingroup BlockClusterTree
 *  \brief The BlockClusterTree class manages block cluster trees for
 *         wavelet constructions.
 */
template <typename ClusterTree> class BlockClusterTree {
public:
  BlockClusterTree() {}
  BlockClusterTree(const ClusterTree &CT) { init(CT); }
  // top level initialisation method
  void init(const ClusterTree &CT, double a_param = 1.01, double dp_param = 1.5,
            double dt_param = 2, double operator_order = 0) {
    tree_data_ = std::make_shared<BlockClusterTreeData>(
        a_param, dp_param, dt_param, operator_order, CT.get_max_wlevel());

    computeBlockClusters(CT, CT);
  }
  //////////////////////////////////////////////////////////////////////////////
  // private members
  //////////////////////////////////////////////////////////////////////////////
private:
  //////////////////////////////////////////////////////////////////////////////
  // private member variables
  //////////////////////////////////////////////////////////////////////////////
  std::shared_ptr<BlockClusterTreeData> tree_data_;
  const ClusterTree *rowCluster_;
  const ClusterTree *colCluster_;
  std::vector<BlockClusterTree> sons_;
};
} // namespace FMCA
#endif
