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
#ifndef FMCA_H2MATRIX_H2CLUSTERTREE_H_
#define FMCA_H2MATRIX_H2CLUSTERTREE_H_

namespace FMCA {

/**
 *  \ingroup H2Matrix
 *  \brief The H2ClusterTree class manages the cluster bases for a given
 *         ClusterTree.
 *
 *         The tree structure from the ClusterTree is replicated here. This
 *         was a design decision as a cluster tree per se is not related to
 *         cluster bases. Also note that we just use pointers to clusters here.
 *         Thus, if the cluster tree is mutated or goes out of scope, we get
 *         dangeling pointers!
 */
template <typename ClusterTree, IndexType Deg>
class H2ClusterTree {
 public:
  typedef typename ClusterTree::value_type value_type;
  enum { dimension = ClusterTree::dimension };
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  H2ClusterTree() {}
  H2ClusterTree(const Eigen::Matrix<value_type, dimension, Eigen::Dynamic> &P,
                const ClusterTree &CT) {
    init(P, CT);
  }

  void init(const Eigen::Matrix<value_type, dimension, Eigen::Dynamic> &P,
            const ClusterTree &CT) {
    // set up the tensor product interpolator
    TP_interp_ = std::make_shared<
        TensorProductInterpolator<value_type, dimension, Deg>>();
    TP_interp_->init();
    // now compute the H2 cluster bases
    computeClusterBases(P, CT);
  }

  void computeClusterBases(
      const Eigen::Matrix<value_type, dimension, Eigen::Dynamic> &P,
      const ClusterTree &CT) {
    cluster_ = &CT;
    if (CT.get_sons().size()) {
      sons_.resize(CT.get_sons().size());
      for (auto i = 0; i < CT.get_sons().size(); ++i) {
        sons_[i].TP_interp_ = TP_interp_;
        sons_[i].computeClusterBases(P, CT.get_sons()[i]);
      }
      // compute transfer matrices
      const eigenMatrix &Xi = TP_interp_->get_Xi();
      for (auto i = 0; i < sons_.size(); ++i) {
        eigenMatrix E(Xi.cols(), Xi.cols());
        for (auto j = 0; j < E.cols(); ++j)
          E.col(j) = TP_interp_->evalLagrangePolynomials(
              (Xi.col(j).array() * sons_[i].cluster_->get_bb().col(2).array() /
                   cluster_->get_bb().col(2).array() +
               (sons_[i].cluster_->get_bb().col(0).array() -
                cluster_->get_bb().col(0).array()) /
                   cluster_->get_bb().col(2).array())
                  .matrix());
        E_.emplace_back(std::move(E));
      }
    }
    V_.resize(TP_interp_->get_Xi().cols(), CT.get_indices().size());
    for (auto i = 0; i < CT.get_indices().size(); ++i) {
      V_.col(i) = TP_interp_->evalLagrangePolynomials(
          ((P.col(CT.get_indices()[i]) - CT.get_bb().col(0)).array() /
           CT.get_bb().col(2).array())
              .matrix());
    }
    // lets have fun checking the transfer matrices...
    eigenMatrix V;
    for (auto i = 0; i < sons_.size(); ++i) {
      V.conservativeResize(sons_[i].V_.rows(), V.cols() + sons_[i].V_.cols());
      V.rightCols(sons_[i].V_.cols()) = E_[i] * sons_[i].V_;
    }
    if (sons_.size()) {
      double nrm = (V - V_).norm() / V_.norm();
      if (nrm > 1e-10)
        std::cout << V_ << std::endl
                  << "-----\n"
                  << V << std::endl
                  << "-----\n"
                  << E_[0] << std::endl
                  << "-----\n"
                  << E_[1] << std::endl
                  << "-----\n"
                  << cluster_->get_bb() << std::endl
                  << "-----\n"

                  << sons_[0].cluster_->get_bb() << std::endl
                  << "-----\n"
                  << sons_[1].cluster_->get_bb() << std::endl
                  << "*************************\n";

      std::cout << nrm << std::endl;
    }

    return;
  }
  //////////////////////////////////////////////////////////////////////////////
 private:
  std::vector<H2ClusterTree> sons_;
  const ClusterTree *cluster_;
  std::shared_ptr<TensorProductInterpolator<value_type, dimension, Deg>>
      TP_interp_;
  std::vector<eigenMatrix> E_;
  eigenMatrix V_;
};

}  // namespace FMCA
#endif
