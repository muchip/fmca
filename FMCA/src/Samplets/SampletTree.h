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

namespace internal {
template <>
struct traits<SampletTreeNode> {
  typedef FloatType value_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
};
}  // namespace internal

struct SampletTreeNode : public SampletTreeNodeBase<SampletTreeNode> {};

namespace internal {
template <>
struct traits<SampletTreeQR> : public traits<ClusterTree> {
  typedef SampletTreeNode node_type;
};
}  // namespace internal

/**
 *  \ingroup Samplets
 *  \brief The SampletTree class manages samplets constructed on a cluster tree.
 */
struct SampletTreeQR : public SampletTreeBase<SampletTreeQR> {
 public:
  typedef typename internal::traits<SampletTreeQR>::value_type value_type;
  typedef typename internal::traits<SampletTreeQR>::node_type node_type;
  typedef typename internal::traits<SampletTreeQR>::eigenMatrix eigenMatrix;
  typedef SampletTreeBase<SampletTreeQR> Base;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  SampletTreeQR() {}
  SampletTreeQR(const eigenMatrix &P, IndexType min_cluster_size = 1,
                IndexType dtilde = 1) {
    init(P, min_cluster_size, dtilde);
    sampletMapper();
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  void init(const eigenMatrix &P, IndexType min_cluster_size = 1,
            IndexType dtilde = 1) {
    // init moment computer
    ClusterTreeBase<SampletTreeQR>::init(P, min_cluster_size);
    SampleMomentComputerQR<SampletTreeQR, MultiIndexSet<TotalDegree>> mom_comp;
    mom_comp.init(P.rows(), dtilde);
    computeSamplets(P, mom_comp);
    //  now map it
    // sampletMapper();
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void sampletTransformMatrix(eigenMatrix &M) {
    M = sampletTransform(M);
    M = sampletTransform(M.transpose());
  }
  //////////////////////////////////////////////////////////////////////////////
  eigenMatrix sampletTransform(const eigenMatrix &data) const {
    assert(is_root() &&
           "sampletTransform needs to be called from the root cluster");
    eigenMatrix retval(data.rows(), data.cols());
    retval.setZero();
    sampletTransformRecursion(data, &retval);
    return retval;
  }

 private:
  template <typename MomentComputer>
  void computeSamplets(const eigenMatrix &P, const MomentComputer &mom_comp) {
    if (nSons()) {
      IndexType offset = 0;
      for (auto i = 0; i < nSons(); ++i) {
        sons(i).computeSamplets(P, mom_comp);
        // the son now has moments, lets grep them...
        eigenMatrix shift = 0.5 * (sons(i).bb().col(0) - bb().col(0) +
                                   sons(i).bb().col(1) - bb().col(1));
        node().mom_buffer_.conservativeResize(
            sons(i).node().mom_buffer_.rows(),
            offset + sons(i).node().mom_buffer_.cols());
        node().mom_buffer_.block(0, offset, sons(i).node().mom_buffer_.rows(),
                                 sons(i).node().mom_buffer_.cols()) =
            mom_comp.shift_matrix(shift) * sons(i).node().mom_buffer_;
        offset += sons(i).node().mom_buffer_.cols();
        // clear moment buffer of the children
        sons(i).node().mom_buffer_.resize(0, 0);
      }
    } else
      // compute cluster basis of the leaf
      node().mom_buffer_ = mom_comp.moment_matrix(P, *this);
    // are there samplets?
    if (mom_comp.mdtilde() < node().mom_buffer_.cols()) {
      Eigen::HouseholderQR<eigenMatrix> qr(node().mom_buffer_.transpose());
      node().Q_ = qr.householderQ();
      node().nscalfs_ = mom_comp.mdtilde();
      node().nsamplets_ = node().Q_.cols() - node().nscalfs_;
      // this is the moment for the dad cluster
      node().mom_buffer_ =
          qr.matrixQR()
              .block(0, 0, mom_comp.mdtilde(), mom_comp.mdtilde2())
              .template triangularView<Eigen::Upper>()
              .transpose();
    } else {
      node().Q_ = eigenMatrix::Identity(node().mom_buffer_.cols(),
                                        node().mom_buffer_.cols());
      node().nscalfs_ = node().mom_buffer_.cols();
      node().nsamplets_ = 0;
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void sampletMapper() {
    assert(is_root() &&
           "sampletMapper needs to be called from the root cluster");
    IndexType i = 0;
    IndexType sum = 0;
    // assign vector start_index to each wavelet cluster
    for (auto &it : *this) {
      it.node().start_index_ = sum;
      it.node().block_id_ = i;
      sum += it.derived().nsamplets();
      if (it.is_root()) sum += it.derived().nscalfs();
    }
    assert(sum == indices().size());
    return;
  }

  eigenMatrix sampletTransformRecursion(const eigenMatrix &data,
                                        eigenMatrix *svec) const {
    eigenMatrix retval(0, 0);
    IndexType scalf_shift = 0;
    if (is_root()) scalf_shift = nscalfs();
    if (nSons()) {
      for (auto i = 0; i < nSons(); ++i) {
        eigenMatrix scalf = sons(i).sampletTransformRecursion(data, svec);
        retval.conservativeResize(retval.rows() + scalf.rows(), data.cols());
        retval.bottomRows(scalf.rows()) = scalf;
      }
    } else {
      retval = data.middleRows(indices_begin(), indices().size());
    }
    if (nsamplets()) {
      svec->middleRows(start_index() + scalf_shift, nsamplets()) =
          Q().rightCols(nsamplets()).transpose() * retval;
      retval = Q().leftCols(nscalfs()).transpose() * retval;
    }
    if (is_root()) svec->middleRows(start_index(), nscalfs()) = retval;
    return retval;
  }

#if 0

  //////////////////////////////////////////////////////////////////////////////
  void inverseSampletTransformMatrix(eigenMatrix &M) {
    for (auto j = 0; j < M.cols(); ++j)
      M.col(j) = inverseSampletTransform(M.col(j));
    for (auto i = 0; i < M.rows(); ++i)
      M.row(i) = inverseSampletTransform(M.row(i));
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
      if (min_id > it->cluster_->get_id()) min_id = it->cluster_->get_id();
      if (max_id < it->cluster_->get_id()) max_id = it->cluster_->get_id();
      std::cout << it->wlevel_ << ")\t"
                << "id: " << it->cluster_->get_id() << std::endl;
    }
    std::cout << "min/max ID: " << min_id << " / " << max_id << std::endl;
    std::cout << "wavelet blocks: " << tree_data_->samplet_list.size()
              << std::endl;
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  IndexType get_nscalfs() const { return nscalfs_; }
  //////////////////////////////////////////////////////////////////////////////
  const eigenMatrix &get_Q() const { return Q_; }
  //////////////////////////////////////////////////////////////////////////////
  const std::vector<SampletTree> &get_sons() const { return sons_; }
  //////////////////////////////////////////////////////////////////////////////
  const ClusterTree *get_cluster() const { return cluster_; }
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<value_type>> get_transformationMatrix() const {
    const IndexType n = cluster_->get_indices().size();
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
    return triplet_list;
  }
  //////////////////////////////////////////////////////////////////////////////
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
  //////////////////////////////////////////////////////////////////////////////
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
      Eigen::Matrix3d bla;
      bla.setZero();
      if (dimension == 2) {
        bla.topRows(2) = cluster_->get_bb();
      }
      // if (dimension == 3) {
      //   bla = cluster_->get_bb();
      // }
      bbvec.push_back(bla);
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
#endif
};
}  // namespace FMCA
#endif
