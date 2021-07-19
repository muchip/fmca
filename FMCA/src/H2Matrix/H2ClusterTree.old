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

#define CHECK_TRANSFER_MATRICES_
namespace FMCA {

template <typename CT>
class H2Matrix;
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
  friend H2Matrix<H2ClusterTree>;

 public:
  typedef typename ClusterTree::value_type value_type;
  enum { dimension = ClusterTree::dimension };
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  H2ClusterTree() {}
  H2ClusterTree(const Eigen::Matrix<value_type, dimension, Eigen::Dynamic> &P,
                ClusterTree &CT) {
    init(P, CT);
  }

  void init(const Eigen::Matrix<value_type, dimension, Eigen::Dynamic> &P,
            ClusterTree &CT) {
    // set up the tensor product interpolator
    TP_interp_ = std::make_shared<
        TensorProductInterpolator<value_type, dimension, Deg>>();
    TP_interp_->init();
    std::cout << "H2 number of polynomials: " << TP_interp_->get_Xi().cols()
              << std::endl;
    // now compute the H2 cluster bases
    computeClusterBases(P, CT);
  }

  void computeClusterBases(
      const Eigen::Matrix<value_type, dimension, Eigen::Dynamic> &P,
      ClusterTree &CT) {
    cluster_ = &CT;
    bool refine = true;
    V_.resize(0, 0);
    E_.clear();
    if (CT.get_sons().size()) {
      // check if all sons have more indices than the number of
      // interpolation points. if not, create a leaf;
      for (auto i = 0; i < CT.get_sons().size(); ++i) {
        if (CT.get_sons()[i].get_indices().size() <
            TP_interp_->get_Xi().cols()) {
          refine = false;
          break;
        }
      }
    }
    if (CT.get_sons().size() && refine) {
      sons_.resize(CT.sons_.size());
      for (auto i = 0; i < CT.sons_.size(); ++i) {
        sons_[i].TP_interp_ = TP_interp_;
        sons_[i].computeClusterBases(P, CT.sons_[i]);
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
    } else {
      //eigen_assert(TP_interp_->get_Xi().cols() <= CT.get_indices().size() &&
      //             "????");
      // compute leaf
      V_.resize(TP_interp_->get_Xi().cols(), CT.get_indices().size());
      for (auto i = 0; i < CT.get_indices().size(); ++i)
        V_.col(i) = TP_interp_->evalLagrangePolynomials(
            ((P.col(CT.get_indices()[i]) - CT.get_bb().col(0)).array() /
             CT.get_bb().col(2).array())
                .matrix());
      // delete children of the cluster tree, as we do not need them for the
      // moment!!!
      cluster_->sons_.clear();
    }
#ifdef CHECK_TRANSFER_MATRICES_
    if (E_.size()) {
      eigenMatrix V;
      V_.resize(TP_interp_->get_Xi().cols(), CT.get_indices().size());
      for (auto i = 0; i < CT.get_indices().size(); ++i)
        V_.col(i) = TP_interp_->evalLagrangePolynomials(
            ((P.col(CT.get_indices()[i]) - CT.get_bb().col(0)).array() /
             CT.get_bb().col(2).array())
                .matrix());

      for (auto i = 0; i < sons_.size(); ++i) {
        V.conservativeResize(sons_[i].V_.rows(), V.cols() + sons_[i].V_.cols());
        V.rightCols(sons_[i].V_.cols()) = E_[i] * sons_[i].V_;
      }
      double nrm = (V - V_).norm() / V_.norm();
      eigen_assert(nrm < 1e-14 && "the H2 cluster basis is faulty");
    }
#endif
    return;
  }

  const std::vector<H2ClusterTree> &get_sons() const { return sons_; }

  const eigenMatrix &get_V() const { return V_; }

  const std::vector<eigenMatrix> &get_E() const { return E_; }

  const ClusterTree &get_cluster() const { return *cluster_; }

  const eigenMatrix &get_Xi() const { return TP_interp_->get_Xi(); }
  //////////////////////////////////////////////////////////////////////////////
 public:
  std::vector<H2ClusterTree> sons_;
  ClusterTree *cluster_;
  std::shared_ptr<TensorProductInterpolator<value_type, dimension, Deg>>
      TP_interp_;
  std::vector<eigenMatrix> E_;
  eigenMatrix V_;
};

}  // namespace FMCA
#endif
