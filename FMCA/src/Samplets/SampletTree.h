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

#define USE_QR_CONSTRUCTION_
namespace FMCA {
template <typename ClusterTree> class SampletTree;

/**
 *  \ingroup Samplets
 *  \brief The SampletTreeData struct keeps all information that should be
 *         available to each node of the samplet tree and is accessed by a
 *         shared pointer to avoid unnecessary data overhead.
 *
 *         In particular, we hace here a levelwise serialisation of the
 *         samplet tree stored in the std::vector samplet_list
 */
template <typename ClusterTree> struct SampletTreeData {
  IndexType max_wlevel_ = 0;
  IndexType dtilde_ = 0;
  IndexType d2tilde_ = 0;
  IndexType m_dtilde_ = 0;
  IndexType m_d2tilde_ = 0;
  typename ClusterTree::value_type orthogonality_threshold_ = 0;
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
  friend class BivariateCompressorH2<SampletTree>;

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
              const ClusterTree &CT, IndexType dtilde,
              value_type orth_thresh = 1e-6) {
    init(P, CT, dtilde);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  void init(const Eigen::Matrix<value_type, dimension, Eigen::Dynamic> &P,
            const ClusterTree &CT, IndexType dtilde,
            value_type orth_thresh = 1e-6) {
    tree_data_ = std::make_shared<SampletTreeData<ClusterTree>>();
#ifdef USE_QR_CONSTRUCTION_
    {
      IndexType d2tlde = 0;
      std::vector<const ClusterTree *> leafs;
      CT.getLeafIterator(leafs);
      IndexType max_cluster_size = 0;
      IndexType min_cluster_size = IndexType(1e10);
      for (const auto &it : leafs) {
        if (max_cluster_size < it->get_indices().size())
          max_cluster_size = it->get_indices().size();
        if (min_cluster_size > it->get_indices().size())
          min_cluster_size = it->get_indices().size();
      }
      std::cout << "min leafsize: " << min_cluster_size
                << " max leafsize: " << max_cluster_size << std::endl;
      IndexType mtlde = 0;
      while (mtlde < min_cluster_size) {
        mtlde += binomialCoefficient(dimension + d2tlde - 1, dimension - 1);
        ++d2tlde;
      }
      --d2tlde;
      std::cout << "internal dtlde: " << d2tlde << " desired dtlde: " << dtilde
                << std::endl;
      tree_data_->dtilde_ = dtilde;
      tree_data_->d2tilde_ = d2tlde > dtilde ? d2tlde : dtilde;
      tree_data_->m_dtilde_ = 0;
      for (auto i = 0; i < dtilde; ++i) {
        tree_data_->m_dtilde_ +=
            binomialCoefficient(dimension + i - 1, dimension - 1);
      }
      tree_data_->idcs.init(tree_data_->d2tilde_ - 1);
      tree_data_->m_d2tilde_ = tree_data_->idcs.get_MultiIndexSet().size();
    }
#else
    {
      tree_data_->dtilde_ = dtilde;
      tree_data_->d2tilde_ = dtilde;
      tree_data_->idcs.init(tree_data_->d2tilde_ - 1);
      tree_data_->m_dtilde_ = tree_data_->idcs.get_MultiIndexSet().size();
      tree_data_->m_d2tilde_ = tree_data_->m_dtilde_;
    }
    tree_data_->orthogonality_threshold_ = orth_thresh;
#endif
    {
      // generate all possible multinomial coefficients from the set of
      // multi indices
      IndexType i = 0;
      IndexType j = 0;
      tree_data_->multinomial_coefficients.resize(tree_data_->m_d2tilde_,
                                                  tree_data_->m_d2tilde_);
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
    computeSamplets(P, CT);
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
    sampletTransformRecursion(data, &retval);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  eigenVector inverseSampletTransform(const eigenVector &data) const {
    assert(!cluster_->get_id() &&
           "sampletTransform needs to be called from the root cluster");
    eigenVector retval(data.size());
    retval.setZero();
    inverseSampletTransformRecursion(data, &retval, nullptr);
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
  IndexType get_nscalfs() const { return nscalfs_; }
  const eigenMatrix &get_Q() const { return Q_; }
  const std::vector<SampletTree> &get_sons() const { return sons_; }
  const ClusterTree *get_cluster() const { return cluster_; }
  //////////////////////////////////////////////////////////////////////////////
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

  template <typename H2ClusterTree>
  void computeMultiscaleClusterBases(const H2ClusterTree &CT) {
    assert(&(CT.get_cluster()) == cluster_);

    if (!wlevel_) {
      // as I do not have a better solution right now, store the interpolation
      // points within the samplet tree
      pXi_ = std::make_shared<eigenMatrix>();
      *pXi_ = CT.get_Xi();
    }
    if (!sons_.size()) {
      V_ = CT.get_V() * Q_;
    } else {
      // compute multiscale cluster bases of sons and update own
      for (auto i = 0; i < sons_.size(); ++i) {
        sons_[i].pXi_ = pXi_;
        sons_[i].computeMultiscaleClusterBases(CT.get_sons()[i]);
      }
      V_.resize(0, 0);
      for (auto i = 0; i < sons_.size(); ++i) {
        V_.conservativeResize(sons_[i].V_.rows(),
                              V_.cols() + sons_[i].nscalfs_);
        V_.rightCols(sons_[i].nscalfs_) =
            CT.get_E()[i] * sons_[i].V_.leftCols(sons_[i].nscalfs_);
      }
      V_ *= Q_;
    }
    return;
  }
  void visualizeCoefficients(const eigenVector &coeffs,
                             const std::string &filename,
                             value_type thresh = 1e-6) {
    std::vector<Eigen::Matrix3d> bbvec;
    std::vector<value_type> cell_values;
    visualizeCoefficientsRecursion(coeffs, bbvec, cell_values, thresh);
    IO::plotBoxes(filename, bbvec, cell_values);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
private:
  //////////////////////////////////////////////////////////////////////////////
  // private methods
  //////////////////////////////////////////////////////////////////////////////
  eigenVector sampletTransformRecursion(const eigenVector &data,
                                        eigenVector *svec) const {
    eigenVector retval(0);
    IndexType scalf_shift = 0;
    if (!wlevel_)
      scalf_shift = nscalfs_;
    if (sons_.size()) {
      for (auto i = 0; i < sons_.size(); ++i) {
        auto scalf = sons_[i].sampletTransformRecursion(data, svec);
        retval.conservativeResize(retval.size() + scalf.size());
        retval.tail(scalf.size()) = scalf;
      }
    } else {
      retval = data.segment(cluster_->get_indices_begin(),
                            cluster_->get_indices().size());
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
                                        eigenVector *fvec, eigenVector *ddata,
                                        IndexType ddata_offset = 0) const {
    eigenVector retval;
    if (!wlevel_) {
      retval = Q_.leftCols(nscalfs_) * data.segment(start_index_, nscalfs_) +
               Q_.rightCols(nsamplets_) *
                   data.segment(start_index_ + nscalfs_, nsamplets_);
    } else {
      retval = Q_.leftCols(nscalfs_) * ddata->segment(ddata_offset, nscalfs_);
      if (nsamplets_)
        retval +=
            Q_.rightCols(nsamplets_) * data.segment(start_index_, nsamplets_);
    }
    if (sons_.size()) {
      ddata_offset = 0;
      for (auto i = 0; i < sons_.size(); ++i) {
        sons_[i].inverseSampletTransformRecursion(data, fvec, &retval,
                                                  ddata_offset);
        ddata_offset += sons_[i].nscalfs_;
      }
    } else {
      fvec->segment(cluster_->get_indices_begin(), retval.size()) = retval;
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void
  computeSamplets(const Eigen::Matrix<value_type, dimension, Eigen::Dynamic> &P,
                  const ClusterTree &CT) {
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
      for (auto i = 0; i < CT.get_sons().size(); ++i) {
        sons_[i].tree_data_ = tree_data_;
        sons_[i].computeSamplets(P, CT.get_sons()[i]);
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
#ifdef USE_QR_CONSTRUCTION_
    // are there wavelets?
    if (mom_buffer_.rows() < mom_buffer_.cols()) {
      Eigen::HouseholderQR<eigenMatrix> qr(mom_buffer_.transpose());
      Q_ = qr.householderQ();
      nscalfs_ = tree_data_->m_dtilde_;
      nsamplets_ = Q_.cols() - nscalfs_;
      // this is the moment for the dad cluster
      mom_buffer_ =
          qr.matrixQR()
              .block(0, 0, tree_data_->m_d2tilde_, tree_data_->m_d2tilde_)
              .template triangularView<Eigen::Upper>()
              .transpose();
      mom_buffer_.conservativeResize(tree_data_->m_d2tilde_, nscalfs_);
    } else {
      Q_ = eigenMatrix::Identity(mom_buffer_.cols(), mom_buffer_.cols());
      nscalfs_ = mom_buffer_.cols();
      nsamplets_ = 0;
    }
#else
    if (mom_buffer_.rows() < mom_buffer_.cols()) {
      // Eigen::JacobiSVD<eigenMatrix> svd(mom_buffer_, Eigen::ComputeFullV);
      Eigen::BDCSVD<eigenMatrix> svd(mom_buffer_,
                                     Eigen::ComputeFullU | Eigen::ComputeFullV);
      nscalfs_ = 0;
      for (nscalfs_ = 0; nscalfs_ < svd.singularValues().size(); ++nscalfs_)
        if (svd.singularValues()(nscalfs_) <
            tree_data_->orthogonality_threshold_)
          break;
      nsamplets_ = mom_buffer_.cols() - nscalfs_;
      Q_ = svd.matrixV();
      mom_buffer_ = (svd.matrixU() * svd.singularValues().asDiagonal())
                        .leftCols(nscalfs_);
    } else {
      Q_ = eigenMatrix::Identity(mom_buffer_.cols(), mom_buffer_.cols());
      nscalfs_ = mom_buffer_.cols();
      nsamplets_ = 0;
    }
#endif
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void sampletMapperRecursion(SampletTree *ST,
                              std::vector<std::vector<SampletTree *>> &mapper) {
    // we always add the root. However, every other cluster is only added if
    // there are indeed wavelets.
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

  void visualizeCoefficientsRecursion(const eigenVector &coeffs,
                                      std::vector<Eigen::Matrix3d> &bbvec,
                                      std::vector<value_type> &cval,
                                      value_type thresh) {
    double color = 0;
    if (!wlevel_) {
      color = coeffs.segment(start_index_, nscalfs_ + nsamplets_)
                  .cwiseAbs()
                  .maxCoeff();
    } else {
      if (nsamplets_)
        color = coeffs.segment(start_index_, nsamplets_).cwiseAbs().maxCoeff();
    }
    if (color > thresh) {
      bbvec.push_back(cluster_->get_bb());
      cval.push_back(color);
    }
    if (sons_.size()) {
      for (auto i = 0; i < sons_.size(); ++i)
        sons_[i].visualizeCoefficientsRecursion(coeffs, bbvec, cval, thresh);
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // private member variables
  //////////////////////////////////////////////////////////////////////////////
private:
  std::vector<SampletTree> sons_;
  std::shared_ptr<SampletTreeData<ClusterTree>> tree_data_;
  const ClusterTree *cluster_;
  std::shared_ptr<eigenMatrix> pXi_;
  eigenMatrix mom_buffer_;
  eigenMatrix Q_;
  eigenMatrix V_;
  IndexType nscalfs_;
  IndexType nsamplets_;
  IndexType wlevel_;
  IndexType start_index_;
  IndexType block_id_;
};
} // namespace FMCA
#endif
