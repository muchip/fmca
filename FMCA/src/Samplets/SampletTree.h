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
template <typename ClusterTree> class SampletTree;

template <typename ClusterTree> struct SampletTreeData {
  IndexType max_wlevel_ = 0;
  IndexType dtilde_ = 0;
  IndexType m_dtilde_ = 0;

  MultiIndexSet<ClusterTree::dimension> idcs;
  std::vector<SampletTree<ClusterTree> *> samplet_list;
  Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic,
                Eigen::Dynamic>
      multinomial_coefficients;
};

/**
 *  \ingroup Samplets
 *  \brief The SampletTree class manages the samplets for a given ClusterTree.
 *
 *         The tree structure from the ClusterTree is replicated here. This
 *         was a design decision as a cluster tree per se is not related to
 *         samplets. Also note that we just use pointers to clusters here. Thus,
 *         if the cluster tree is mutated or goes out of scope, we get dangeling
 *         pointers!
 */
template <typename ClusterTree> class SampletTree {
  friend class BivariateCompressor<SampletTree>;

public:
  typedef typename ClusterTree::value_type value_type;
  enum { dimension = ClusterTree::dimension };
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1> eigenVector;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  SampletTree() {}
  SampletTree(const ClusterTree &CT, IndexType dtilde = 1) { init(CT, dtilde); }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  void init(const ClusterTree &CT, IndexType dtilde = 1) {
    tree_data_ = std::make_shared<SampletTreeData<ClusterTree>>();
    tree_data_->idcs.init(dtilde - 1);
    tree_data_->dtilde_ = dtilde;
    tree_data_->m_dtilde_ = tree_data_->idcs.get_MultiIndexSet().size();
    {
      // generate all possible multinomial coefficients from the set of
      // multi indices
      IndexType i = 0;
      IndexType j = 0;
      tree_data_->multinomial_coefficients.resize(tree_data_->m_dtilde_,
                                                  tree_data_->m_dtilde_);
      for (auto &beta : tree_data_->idcs.get_MultiIndexSet()) {
        for (auto &alpha : tree_data_->idcs.get_MultiIndexSet()) {
          tree_data_->multinomial_coefficients(j, i) =
              multinomialCoefficient<ClusterTree::dimension>(alpha, beta);
          ++j;
        }
        ++i;
        j = 0;
      }
    }
    // compute the samplet basis
    computeSamplets(CT);
    // now map it
    sampletMapper();
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void sampletTransformMatrix(eigenMatrix &M) {
    for (auto j = 0; j < M.cols(); ++j)
      M.col(j) = sampletTransform(M.col(j));
    for (auto i = 0; i < M.rows(); ++i)
      M.row(i) = sampletTransform(M.row(i));
  }
  //////////////////////////////////////////////////////////////////////////////
  void inverseSampletTransformMatrix(eigenMatrix &M) {
    for (auto j = 0; j < M.cols(); ++j)
      M.col(j) = inverseSampletTransform(M.col(j));
    for (auto i = 0; i < M.rows(); ++i)
      M.row(i) = inverseSampletTransform(M.row(i));
  }
  //////////////////////////////////////////////////////////////////////////////
  eigenVector sampletTransform(const eigenVector &data) const {
    assert(!cluster_->get_id() &&
           "sampletTransform needs to be called from the root cluster");
    eigenVector retval(data.size());
    retval.setZero();
    sampletTransformRecursion(data, 0, &retval);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  eigenVector inverseSampletTransform(const eigenVector &data) const {
    assert(!cluster_->get_id() &&
           "sampletTransform needs to be called from the root cluster");
    eigenVector retval(data.size());
    retval.setZero();
    inverseSampletTransformRecursion(data, 0, &retval, nullptr);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  void basisInfo() {
    IndexType cur_level = 0;
    IndexType max_id = 0;
    IndexType min_id = IndexType(1e10);

    for (auto it : tree_data_->samplet_list) {
      if (cur_level != it->wlevel_) {
        cur_level = it->wlevel_;
        std::cout << "min/max ID: " << min_id << " / " << max_id << std::endl;
        max_id = it->cluster_->get_id();
        min_id = it->cluster_->get_id();
      }
      if (min_id > it->cluster_->get_id())
        min_id = it->cluster_->get_id();
      if (max_id < it->cluster_->get_id())
        max_id = it->cluster_->get_id();
      std::cout << it->wlevel_ << ")\t"
                << "id: " << it->cluster_->get_id() << std::endl;
    }
    std::cout << "min/max ID: " << min_id << " / " << max_id << std::endl;
    std::cout << "wavelet blocks: " << tree_data_->samplet_list.size()
              << std::endl;
    return;
  }
  Eigen::SparseMatrix<value_type> get_transformationMatrix() const {
    const IndexType n = cluster_->get_indices().size();
    Eigen::SparseMatrix<value_type> Tmat(n, n);
    eigenVector buffer(n);
    std::vector<Eigen::Triplet<value_type>> triplet_list;
    for (auto j = 0; j < n; ++j) {
      buffer.setZero();
      buffer = sampletTransform(
          Eigen::Matrix<value_type, Eigen::Dynamic, 1>::Unit(n, j));
      for (auto i = 0; i < buffer.size(); ++i)
        if (abs(buffer(i)) > 1e-14)
          triplet_list.emplace_back(
              Eigen::Triplet<value_type>(i, j, buffer(i)));
    }
    Tmat.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return Tmat;
  }

  //////////////////////////////////////////////////////////////////////////////
private:
  //////////////////////////////////////////////////////////////////////////////
  // private methods
  //////////////////////////////////////////////////////////////////////////////
  eigenVector sampletTransformRecursion(const eigenVector &data,
                                        IndexType offset,
                                        eigenVector *svec) const {
    eigenVector retval(0);
    IndexType scalf_shift = 0;
    if (!wlevel_)
      scalf_shift = Q_.cols() - nsamplets_;
    if (sons_.size()) {
      for (auto i = 0; i < sons_.size(); ++i) {
        auto scalf = sons_[i].sampletTransformRecursion(data, offset, svec);
        retval.conservativeResize(retval.size() + scalf.size());
        retval.tail(scalf.size()) = scalf;
        offset += sons_[i].cluster_->get_indices().size();
      }
    } else {
      retval = data.segment(offset, cluster_->get_indices().size());
    }
    if (nsamplets_) {
      svec->segment(start_index_ + scalf_shift, nsamplets_) =
          Q_.rightCols(nsamplets_).transpose() * retval;
      retval = Q_.leftCols(nscalfs_).transpose() * retval;
    }
    if (!wlevel_)
      svec->segment(start_index_, nscalfs_) = retval;
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  void inverseSampletTransformRecursion(const eigenVector &data,
                                        IndexType offset, eigenVector *fvec,
                                        eigenVector *ddata,
                                        IndexType ddata_offset = 0) const {
    eigenVector retval;
    if (!wlevel_) {
      retval = Q_.leftCols(nscalfs_) * data.segment(start_index_, nscalfs_) +
               Q_.rightCols(nsamplets_) *
                   data.segment(start_index_ + nscalfs_, nsamplets_);
    } else {
#if 0
      std::cout << "-----------\n";
      std::cout << "os: " << ddata_offset << " ds: " << ddata->size()
                << std::endl;
      std::cout << "id: " << block_id_ << " wlvl: " << wlevel_
                << " nsf: " << nscalfs_ << " ns: " << nsamplets_
                << " cs: " << cluster_->get_indices().size()
                << " sns: " << sons_.size() << std::endl;
      std::cout << "Q:\n";
      std::cout << Q_ << std::endl;
      eigenVector vec = ddata->segment(ddata_offset, nscalfs_);
      std::cout << "vec:\n";
      std::cout << vec << std::endl;
#endif
      retval = Q_.leftCols(nscalfs_) * ddata->segment(ddata_offset, nscalfs_);
      if (nsamplets_)
        retval +=
            Q_.rightCols(nsamplets_) * data.segment(start_index_, nsamplets_);
    }
    if (sons_.size()) {
      ddata_offset = 0;
      for (auto i = 0; i < sons_.size(); ++i) {
#if 0
        std::cout << " sid: " << i << " cs: " << cluster_->get_indices().size()
                  << " Q: " << Q_.rows() << "/" << Q_.cols() << std::endl;
        std::cout << retval << std::endl;
        std::cout << "s0: " << sons_[0].nscalfs_ << " s1: " << sons_[1].nscalfs_
                  << std::endl;
#endif
        sons_[i].inverseSampletTransformRecursion(data, offset, fvec, &retval,
                                                  ddata_offset);
        offset += sons_[i].cluster_->get_indices().size();
        ddata_offset += sons_[i].nscalfs_;
      }
    } else {
      fvec->segment(offset, retval.size()) = retval;
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void computeSamplets(const ClusterTree &CT) {
    cluster_ = &CT;
    // the computation of the samplet level is a bit cumbersome as we have to
    // account for empty clusters and clusters with a single point here.
    if (CT.get_indices().size()) {
      if (CT.get_indices().size() == 1)
        wlevel_ = CT.get_tree_data().max_level_;
      else {
        int wlevel = ceil(-log(CT.get_bb().col(2).norm() /
                               CT.get_tree_data().geometry_diam_) /
                          log(2));
        wlevel_ = wlevel > 0 ? wlevel : 0;
      }
    } else
      wlevel_ = CT.get_tree_data().max_level_ + 1;

    if (tree_data_->max_wlevel_ < wlevel_)
      tree_data_->max_wlevel_ = wlevel_;
    if (CT.get_sons().size()) {
      sons_.resize(CT.get_sons().size());
      IndexType offset = 0;
      for (auto i = 0; i != CT.get_sons().size(); ++i) {
        sons_[i].tree_data_ = tree_data_;
        sons_[i].computeSamplets(CT.get_sons()[i]);
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
      mom_buffer_ = momentComputer<ClusterTree>(*cluster_, tree_data_->idcs);
    }
    // are there wavelets?
    if (mom_buffer_.rows() < mom_buffer_.cols()) {
      Eigen::HouseholderQR<eigenMatrix> qr(mom_buffer_.transpose());
      Q_ = qr.householderQ();
      nscalfs_ = tree_data_->m_dtilde_;
      nsamplets_ = Q_.cols() - nscalfs_;
      // this is the moment for the dad cluster
      mom_buffer_ =
          qr.matrixQR()
              .block(0, 0, tree_data_->m_dtilde_, tree_data_->m_dtilde_)
              .template triangularView<Eigen::Upper>()
              .transpose();
    } else {
      Q_ = eigenMatrix::Identity(mom_buffer_.cols(), mom_buffer_.cols());
      nscalfs_ = mom_buffer_.cols();
      nsamplets_ = 0;
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void sampletMapperRecursion(SampletTree *ST,
                              std::vector<std::vector<SampletTree *>> &mapper) {
    // we always add the root. However, every other cluster is only added if
    // there are indeed wavelets.
    if (!ST->wlevel_ || ST->nsamplets_)
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
    {
      // this method traverses the samplet tree and stores them in a levelwise
      // ordering  [l_0|l_1|l_2|...]
      std::vector<std::vector<SampletTree *>> mapper(max_wlevel + 1);
      sampletMapperRecursion(this, mapper);
      tree_data_->samplet_list.clear();
      for (auto it : mapper)
        tree_data_->samplet_list.insert(tree_data_->samplet_list.end(),
                                        it.begin(), it.end());
    }
    IndexType sum = 0;
    // assign vector start_index to each wavelet cluster
    for (auto i = 0; i < tree_data_->samplet_list.size(); ++i) {
      tree_data_->samplet_list[i]->start_index_ = sum;
      tree_data_->samplet_list[i]->block_id_ = i;
      sum += tree_data_->samplet_list[i]->nsamplets_;
      if (tree_data_->samplet_list[i]->wlevel_ == 0)
        sum += tree_data_->samplet_list[i]->Q_.cols() -
               tree_data_->samplet_list[i]->nsamplets_;
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // private member variables
  //////////////////////////////////////////////////////////////////////////////
  std::vector<SampletTree> sons_;
  std::shared_ptr<SampletTreeData<ClusterTree>> tree_data_;
  const ClusterTree *cluster_;
  Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> mom_buffer_;
  Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> Q_;
  IndexType nscalfs_;
  IndexType nsamplets_;
  IndexType wlevel_;
  IndexType start_index_;
  IndexType block_id_;
}; // namespace FMCA
} // namespace FMCA
#endif