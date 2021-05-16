// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_SAMPLETS_SAMPLETTREE_H_
#define FMCA_SAMPLETS_SAMPLETTREE_H_

namespace FMCA {

template <typename ValueType, IndexType Dim> struct SampletTreeData {
  IndexType max_wlevel_ = 0;
  MultiIndexSet<Dim> idcs;
  Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>
      multinomial_coefficients;
};

/**
 *  \ingroup SampletTree
 *  \brief The SampletTree class manages the samplets for a given ClusterTree.
 *
 *         The tree structure from the ClusterTree is replicated here. This
 *         was a design decision as a cluster tree per se is not related to
 *         samplets. Also note that we just use pointers to clusters here. Thus,
 *         if the cluster tree is mutated or goes out of scope, we get dangeling
 *         pointers!
 */
template <typename ClusterTree> class SampletTree {

public:
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  SampletTree() {}
  SampletTree(const Eigen::Matrix<typename ClusterTree::value_type,
                                  ClusterTree::dimension, Eigen::Dynamic> &P,
              const ClusterTree &CT, IndexType dtilde = 1) {
    init(P, CT, dtilde);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  void init(const Eigen::Matrix<typename ClusterTree::value_type,
                                ClusterTree::dimension, Eigen::Dynamic> &P,
            const ClusterTree &CT, IndexType dtilde = 1) {
    tree_data_ =
        std::make_shared<SampletTreeData<typename ClusterTree::value_type,
                                         ClusterTree::dimension>>();
    tree_data_->idcs.init(dtilde);
    auto set = tree_data_->idcs.get_MultiIndexSet();
    for (auto i : set) {
      for (auto j : i)
        std::cout << j << " ";
      std::cout << std::endl;
    }
    IndexType i = 0;
    IndexType j = 0;
    tree_data_->multinomial_coefficients.resize(set.size(), set.size());
    for (auto alpha : set) {
      for (auto beta : set) {
        tree_data_->multinomial_coefficients(j, i) =
            multinomialCoefficient<ClusterTree::dimension>(alpha, beta);
        ++j;
      }
      ++i;
      j = 0;
    }
    std::cout << tree_data_->multinomial_coefficients << std::endl;

    computeSamplets(P, CT, dtilde);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  unsigned int get_max_wlevel() const { return tree_data_->max_wlevel_; }
  //////////////////////////////////////////////////////////////////////////////
private:
  //////////////////////////////////////////////////////////////////////////////
  // private methods
  //////////////////////////////////////////////////////////////////////////////
  void computeSamplets(
      const Eigen::Matrix<typename ClusterTree::value_type,
                          ClusterTree::dimension, Eigen::Dynamic> &P,
      const ClusterTree &CT, IndexType dtilde) {
    cluster_ = &CT;
    int wlevel =
        -log(CT.get_bb().col(2).norm() / CT.get_tree_data().geometry_diam_) /
        log(2);
    wlevel_ = wlevel > 0 ? wlevel : 0;
    if (CT.get_sons().size()) {
      sons_.resize(CT.get_sons().size());

      for (auto i = 0; i != CT.get_sons().size(); ++i) {
        sons_[i].tree_data_ = tree_data_;
        sons_[i].computeSamplets(P, CT.get_sons()[i], dtilde);
      }
      if (!sons_[0].sons_.size()) {
        std::cout << "---------------------------\n";
        Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic,
                      Eigen::Dynamic>
            Mom = momentComputer<ClusterTree>(P, *cluster_, tree_data_->idcs);
        std::cout << 0.5 *
                         (cluster_->get_bb().col(0) + cluster_->get_bb().col(1))
                  << std::endl;
        std::cout << Mom << std::endl << std::endl;
        for (auto i = 0; i != CT.get_sons().size(); ++i) {
          Mom = momentComputer<ClusterTree>(P, *(sons_[i].cluster_),
                                            tree_data_->idcs);
          momentShifter<ClusterTree>(
              Mom,
              0.5 * (cluster_->get_bb().col(0) + cluster_->get_bb().col(1)),
              0.5 * (sons_[i].cluster_->get_bb().col(0) +
                     sons_[i].cluster_->get_bb().col(1)),
              tree_data_->idcs, tree_data_->multinomial_coefficients);
          std::cout << 0.5 * (sons_[i].cluster_->get_bb().col(0) +
                              sons_[i].cluster_->get_bb().col(1))
                    << std::endl;
          std::cout << Mom << std::endl << std::endl;
        }
      }
      // here we should now compute the Cluster basis from the children's
      // cluster bases
    } else {
      // compute cluster basis of the leaf
      Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic,
                    Eigen::Dynamic>
          Mom = momentComputer<ClusterTree>(P, *cluster_, tree_data_->idcs);
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // private member variables
  //////////////////////////////////////////////////////////////////////////////
  std::vector<SampletTree> sons_;
  std::shared_ptr<
      SampletTreeData<typename ClusterTree::value_type, ClusterTree::dimension>>
      tree_data_;
  const ClusterTree *cluster_;
  IndexType wlevel_;
};
} // namespace FMCA
#endif
