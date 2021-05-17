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
  IndexType m_dtilde_ = 0;
  MultiIndexSet<Dim> idcs;
  std::vector<IndexType> samplet_mapper;
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
  typedef typename ClusterTree::value_type value_type;
  enum { dimension = ClusterTree::dimension };
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1> eigenVector;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  SampletTree() {}
  SampletTree(const Eigen::Matrix<value_type, dimension, Eigen::Dynamic> &P,
              const ClusterTree &CT, IndexType dtilde = 1) {
    init(P, CT, dtilde);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  void init(const Eigen::Matrix<value_type, dimension, Eigen::Dynamic> &P,
            const ClusterTree &CT, IndexType dtilde = 1) {
    tree_data_ = std::make_shared<SampletTreeData<value_type, dimension>>();
    tree_data_->idcs.init(dtilde);
    tree_data_->m_dtilde_ = tree_data_->idcs.get_MultiIndexSet().size();
    {
      // generate all possible multinomial coefficients from the set of
      // multi indices
      IndexType i = 0;
      IndexType j = 0;
      tree_data_->multinomial_coefficients.resize(tree_data_->m_dtilde_,
                                                  tree_data_->m_dtilde_);
      for (auto beta : tree_data_->idcs.get_MultiIndexSet()) {
        for (auto alpha : tree_data_->idcs.get_MultiIndexSet()) {
          tree_data_->multinomial_coefficients(j, i) =
              multinomialCoefficient<ClusterTree::dimension>(alpha, beta);
          ++j;
        }
        ++i;
        j = 0;
      }
    }
    // compute the samplet basis
    computeSamplets(P, CT, dtilde);
    // now map it
    sampletMapper();
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  unsigned int get_max_wlevel() const { return tree_data_->max_wlevel_; }
  //////////////////////////////////////////////////////////////////////////////
private:
  //////////////////////////////////////////////////////////////////////////////
  // private methods
  //////////////////////////////////////////////////////////////////////////////
  void
  computeSamplets(const Eigen::Matrix<value_type, dimension, Eigen::Dynamic> &P,
                  const ClusterTree &CT, IndexType dtilde) {
    cluster_ = &CT;
    int wlevel = ceil(
        -log(CT.get_bb().col(2).norm() / CT.get_tree_data().geometry_diam_) /
        log(2));
    wlevel_ = wlevel > 0 ? wlevel : 0;
    if (tree_data_->max_wlevel_ < wlevel_)
      tree_data_->max_wlevel_ = wlevel_;
    if (CT.get_sons().size()) {
      sons_.resize(CT.get_sons().size());
      IndexType offset = 0;
      for (auto i = 0; i != CT.get_sons().size(); ++i) {
        sons_[i].tree_data_ = tree_data_;
        sons_[i].computeSamplets(P, CT.get_sons()[i], dtilde);
        // the son now has moments, lets grep them...
        eigenVector shift =
            0.5 *
            (sons_[i].cluster_->get_bb().col(0) - cluster_->get_bb().col(0) +
             sons_[i].cluster_->get_bb().col(1) - cluster_->get_bb().col(1));
        eigenMatrix T = momentShifter<ClusterTree>(
            shift, tree_data_->idcs, tree_data_->multinomial_coefficients);
        mom_buffer_.conservativeResize(sons_[i].mom_buffer_.rows(),
                                       offset + sons_[i].mom_buffer_.cols());
        mom_buffer_.block(0, offset, sons_[i].mom_buffer_.rows(),
                          sons_[i].mom_buffer_.cols()) =
            T * sons_[i].mom_buffer_;
        offset += sons_[i].mom_buffer_.cols();
        // clear moment buffer of the children
        sons_[i].mom_buffer_.resize(0, 0);
      }
    } else {
      // compute cluster basis of the leaf
      mom_buffer_ = momentComputer<ClusterTree>(P, *cluster_, tree_data_->idcs);
    }
    // are there wavelets?
    if (mom_buffer_.rows() < mom_buffer_.cols()) {
      Eigen::HouseholderQR<eigenMatrix> qr(mom_buffer_.transpose());
      Q_ = qr.householderQ();
      // this is the moment for the dad cluster
      mom_buffer_ =
          qr.matrixQR()
              .block(0, 0, tree_data_->m_dtilde_, tree_data_->m_dtilde_)
              .template triangularView<Eigen::Upper>()
              .transpose();
    } else {
      Q_.resize(0, 0);
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void sampletMapperRecursion(SampletTree *ST,
                              std::vector<std::vector<SampletTree *>> &mapper) {
    mapper[ST->wlevel_].push_back(ST);
    if (ST->sons_.size())
      for (auto i = 0; i < ST->sons_.size(); ++i)
        sampletMapperRecursion(&(ST->sons_[i]), mapper);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void sampletMapper() {
    assert(!cluster_->get_id() &&
           "sampletMapper needs to be called from the root cluster");
    IndexType n_pts = cluster_->get_indices().size();
    IndexType max_wlevel = tree_data_->max_wlevel_;
    IndexType max_cluster_id = cluster_->get_tree_data().max_id_;
    std::vector<std::vector<SampletTree *>> mapper;
    mapper.resize(max_wlevel + 1);
    // this method traverses the samplet tree and stores them in a rowwise
    // ordering
    sampletMapperRecursion(this, mapper);
    // next, we serialize the the samplet tree by storing coefficents
    // according to [l_0|l_1|l_2|...]
    IndexType sum = 0;
    tree_data_->samplet_mapper.resize(max_cluster_id + 1);
    std::fill(tree_data_->samplet_mapper.begin(),
              tree_data_->samplet_mapper.end(), n_pts);
    for (auto i = 0; i < mapper.size(); ++i)
      for (auto j = 0; j < mapper[i].size(); ++j) {
        tree_data_->samplet_mapper[mapper[i][j]->cluster_->get_id()] = sum;
        if (mapper[i][j]->Q_.cols()) {
          sum += mapper[i][j]->Q_.cols() - tree_data_->m_dtilde_;
        }
      }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // private member variables
  //////////////////////////////////////////////////////////////////////////////
  std::vector<SampletTree> sons_;
  std::shared_ptr<SampletTreeData<value_type, dimension>> tree_data_;
  const ClusterTree *cluster_;
  Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> mom_buffer_;
  Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> Q_;
  IndexType wlevel_;
}; // namespace FMCA
} // namespace FMCA
#endif
