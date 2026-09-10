// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
////////////////////////////////////////////////////////////////////////////////
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/Tictoc.h>

#include <FMCA/Clustering>
#include <FMCA/CovarianceKernel>
#include <FMCA/H2Matrix>
#include <FMCA/LowRankApproximation>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
namespace py = pybind11;
// Samplets
using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using H2SampletTreeRP = FMCA::H2SampletTree<FMCA::RandomProjectionTree>;
// H2Matrix
using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;
////////////////////////////////////////////////////////////////////////////////

using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
////////////////////////////////////////////////////////////////////////////////
//  helpers shared by the samplet tree wrappers
////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief returns the leaves of the adaptive tree determined by the second
 *         Binev-DeVore algorithm, i.e. the active clusters without active
 *         sons. These clusters form a partition of the point cloud. thres is
 *         understood relative to the energy of the data.
 **/
template <typename Derived>
std::vector<const Derived *> adaptiveTreeLeafs(const Derived &ST,
                                               const FMCA::Vector &data,
                                               const FMCA::Scalar thres) {
  const FMCA::Vector tdata = ST.sampletTransform(ST.toClusterOrder(data));
  const std::vector<const Derived *> active_tree =
      FMCA::adaptiveTreeSearch(ST, tdata, thres * data.squaredNorm());
  std::vector<const Derived *> leafs;
  for (const auto &node : active_tree) {
    if (node == nullptr || !node->block_size()) continue;
    bool has_active_son = false;
    for (FMCA::Index i = 0; i < node->nSons(); ++i)
      if (node->sons(i).block_size() &&
          active_tree[node->sons(i).block_id()] != nullptr)
        has_active_son = true;
    if (!has_active_son) leafs.push_back(node);
  }
  return leafs;
}

/**
 *  \brief returns all non-empty clusters on a given level of the tree
 **/
template <typename Derived>
std::vector<const Derived *> levelClusters(const Derived &ST,
                                           const FMCA::Index lvl) {
  std::vector<const Derived *> retval;
  for (const auto &it : ST)
    if (it.level() == lvl && it.block_size())
      retval.push_back(std::addressof(it));
  return retval;
}

/**
 *  \brief labels each point by the index of the cluster in clusters it
 *         belongs to. Points in none of the clusters are labelled -1.
 **/
template <typename Derived>
Eigen::VectorXi clusterLabels(const std::vector<const Derived *> &clusters,
                              const FMCA::Index npts) {
  Eigen::VectorXi retval(npts);
  retval.setConstant(-1);
  for (FMCA::Index c = 0; c < clusters.size(); ++c)
    for (FMCA::Index i = 0; i < clusters[c]->block_size(); ++i)
      retval(clusters[c]->indices()[i]) = int(c);
  return retval;
}

/**
 *  \brief bounding boxes of a list of clusters as a (2 * dim) x nclusters
 *         matrix, each column stacking [lower corner; upper corner]
 **/
template <typename Derived>
FMCA::Matrix clusterBoxes(const std::vector<const Derived *> &clusters) {
  if (!clusters.size()) return FMCA::Matrix(0, 0);
  const FMCA::Index dim = clusters[0]->bb().rows();
  FMCA::Matrix retval(2 * dim, clusters.size());
  for (FMCA::Index c = 0; c < clusters.size(); ++c) {
    retval.col(c).head(dim) = clusters[c]->bb().col(0);
    retval.col(c).tail(dim) = clusters[c]->bb().col(1);
  }
  return retval;
}

/**
 *  \brief the samplet transform written out as a sparse matrix T, such that
 *         T * data equals sampletTransform(data) for cluster ordered data
 **/
template <typename Derived>
Eigen::SparseMatrix<FMCA::Scalar> sampletTransformationMatrix(
    const Derived &ST) {
  const std::vector<FMCA::Triplet> trips = ST.transformationMatrixTriplets2();
  Eigen::SparseMatrix<FMCA::Scalar> retval(ST.block_size(), ST.block_size());
  retval.setFromTriplets(trips.begin(), trips.end());
  return retval;
}
/**
 *  \brief wrapper class for a samplet tree (for convenience, we only use H2
 *         trees)
 *
 **/
struct pySampletTree {
  pySampletTree() {};
  pySampletTree(const FMCA::Matrix &P, FMCA::Index dtilde) {
    dtilde_ = dtilde > 0 ? dtilde : 1;
    p_ = 2 * (dtilde_ - 1);
    const Moments mom(P, p_);
    const SampletMoments samp_mom(P, dtilde - 1);
    ST_.init(mom, samp_mom, 0, P);
    cluster_map_.resize(P.cols());

    for (const auto &it : ST_)
      if (it.is_root())
        for (FMCA::Index i = 0; i < it.nscalfs() + it.nsamplets(); ++i)
          cluster_map_[it.start_index() + i] = std::addressof(it);
      else
        for (FMCA::Index i = 0; i < it.nsamplets(); ++i)
          cluster_map_[it.start_index() + i] = std::addressof(it);
  };

  FMCA::iVector indices() {
    return Eigen::Map<const FMCA::iVector>(ST_.indices(), ST_.block_size());
  }
  FMCA::iVector levels() {
    std::vector<FMCA::Index> lvl = FMCA::internal::sampletLevelMapper(ST_);
    return Eigen::Map<const FMCA::iVector>(lvl.data(), lvl.size());
  }

  FMCA::iVector coeff2indices(FMCA::Index i) {
    const H2SampletTree &node = *(cluster_map_[i]);
    return Eigen::Map<const FMCA::iVector>(node.indices(), node.block_size());
  }

  FMCA::Matrix toClusterOrder(const FMCA::Matrix &mat) const {
    return ST_.toClusterOrder(mat);
  }

  FMCA::Matrix toNaturalOrder(const FMCA::Matrix &mat) const {
    return ST_.toNaturalOrder(mat);
  }

  FMCA::Matrix sampletTransform(const FMCA::Matrix &data) const {
    return ST_.sampletTransform(data);
  }

  FMCA::Matrix inverseSampletTransform(const FMCA::Matrix &data) const {
    return ST_.inverseSampletTransform(data);
  }

  FMCA::Matrix sampletTransformMatrix(const FMCA::Matrix &M) const {
    const FMCA::Matrix buf = ST_.sampletTransform(M);
    return ST_.sampletTransform(buf.transpose()).transpose();
  }

  FMCA::Matrix inverseSampletTransformMatrix(const FMCA::Matrix &M) const {
    const FMCA::Matrix buf = ST_.inverseSampletTransform(M);
    return ST_.inverseSampletTransform(buf.transpose()).transpose();
  }

  Eigen::SparseMatrix<FMCA::Scalar> transformationMatrix() const {
    return sampletTransformationMatrix(ST_);
  }

  Eigen::VectorXi adaptiveTreeLeafPartition(const FMCA::Vector &data,
                                            FMCA::Scalar thres) const {
    return clusterLabels(adaptiveTreeLeafs(ST_, data, thres), ST_.block_size());
  }

  FMCA::Matrix adaptiveTreeLeafBoxes(const FMCA::Vector &data,
                                     FMCA::Scalar thres) const {
    return clusterBoxes(adaptiveTreeLeafs(ST_, data, thres));
  }

  Eigen::VectorXi levelLabels(FMCA::Index lvl) const {
    return clusterLabels(levelClusters(ST_, lvl), ST_.block_size());
  }

  FMCA::Matrix levelBoxes(FMCA::Index lvl) const {
    return clusterBoxes(levelClusters(ST_, lvl));
  }

  FMCA::Index dim() const { return ST_.bb().rows(); }
  FMCA::Index npts() const { return ST_.block_size(); }
  FMCA::Index dtilde() const { return dtilde_; }
  FMCA::Index nscalfs() const { return ST_.nscalfs(); }
  FMCA::Index nclusters() const {
    return std::distance(ST_.begin(), ST_.end());
  }

  H2SampletTree ST_;
  FMCA::Index p_;
  FMCA::Index dtilde_;
  std::vector<const H2SampletTree *> cluster_map_;
};

/**
 *  \brief wrapper class for a samplet tree based on a random projection tree
 *  (for convenience, we only use H2
 *         trees)
 *
 **/
struct pySampletTreeRP {
  pySampletTreeRP() {};
  pySampletTreeRP(const FMCA::Matrix &P, FMCA::Index dtilde,
                  FMCA::Index seed = 0) {
    dtilde_ = dtilde > 0 ? dtilde : 1;
    p_ = 2 * (dtilde_ - 1);
    const Moments mom(P, p_);
    const SampletMoments samp_mom(P, dtilde - 1);
    srand(seed);
    ST_.init(mom, samp_mom, 10, P);
    cluster_map_.resize(P.cols());
    for (const auto &it : ST_)
      if (it.is_root())
        for (FMCA::Index i = 0; i < it.nscalfs() + it.nsamplets(); ++i)
          cluster_map_[it.start_index() + i] = std::addressof(it);
      else
        for (FMCA::Index i = 0; i < it.nsamplets(); ++i)
          cluster_map_[it.start_index() + i] = std::addressof(it);
  };

  FMCA::iVector indices() {
    return Eigen::Map<const FMCA::iVector>(ST_.indices(), ST_.block_size());
  }
  FMCA::iVector levels() {
    std::vector<FMCA::Index> lvl = FMCA::internal::sampletLevelMapper(ST_);
    return Eigen::Map<const FMCA::iVector>(lvl.data(), lvl.size());
  }

  FMCA::Matrix toClusterOrder(const FMCA::Matrix &mat) const {
    return ST_.toClusterOrder(mat);
  }

  FMCA::Matrix toNaturalOrder(const FMCA::Matrix &mat) const {
    return ST_.toNaturalOrder(mat);
  }

  FMCA::iVector coeff2indices(FMCA::Index i) {
    const H2SampletTreeRP &node = *(cluster_map_[i]);
    return Eigen::Map<const FMCA::iVector>(node.indices(), node.block_size());
  }

  FMCA::Matrix sampletTransform(const FMCA::Matrix &data) const {
    return ST_.sampletTransform(data);
  }

  FMCA::Matrix inverseSampletTransform(const FMCA::Matrix &data) const {
    return ST_.inverseSampletTransform(data);
  }

  FMCA::Matrix sampletTransformMatrix(const FMCA::Matrix &M) const {
    const FMCA::Matrix buf = ST_.sampletTransform(M);
    return ST_.sampletTransform(buf.transpose()).transpose();
  }

  FMCA::Matrix inverseSampletTransformMatrix(const FMCA::Matrix &M) const {
    const FMCA::Matrix buf = ST_.inverseSampletTransform(M);
    return ST_.inverseSampletTransform(buf.transpose()).transpose();
  }

  Eigen::SparseMatrix<FMCA::Scalar> transformationMatrix() const {
    return sampletTransformationMatrix(ST_);
  }

  Eigen::VectorXi adaptiveTreeLeafPartition(const FMCA::Vector &data,
                                            FMCA::Scalar thres) const {
    return clusterLabels(adaptiveTreeLeafs(ST_, data, thres), ST_.block_size());
  }

  FMCA::Matrix adaptiveTreeLeafBoxes(const FMCA::Vector &data,
                                     FMCA::Scalar thres) const {
    return clusterBoxes(adaptiveTreeLeafs(ST_, data, thres));
  }

  Eigen::VectorXi levelLabels(FMCA::Index lvl) const {
    return clusterLabels(levelClusters(ST_, lvl), ST_.block_size());
  }

  FMCA::Matrix levelBoxes(FMCA::Index lvl) const {
    return clusterBoxes(levelClusters(ST_, lvl));
  }

  FMCA::Index dim() const { return ST_.bb().rows(); }
  FMCA::Index npts() const { return ST_.block_size(); }
  FMCA::Index dtilde() const { return dtilde_; }
  FMCA::Index nscalfs() const { return ST_.nscalfs(); }
  FMCA::Index nclusters() const {
    return std::distance(ST_.begin(), ST_.end());
  }

  H2SampletTreeRP ST_;
  FMCA::Index p_;
  FMCA::Index dtilde_;
  std::vector<const H2SampletTreeRP *> cluster_map_;
};

////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief wrapper class for an H2Matrix
 *
 **/
struct pyH2Matrix {
  pyH2Matrix(const FMCA::CovarianceKernel &ker, const FMCA::Matrix &Pr,
             const FMCA::Matrix &Pc, const FMCA::Index p = 3,
             const FMCA::Scalar eta = 0.8)
      : Pr_(Pr), Pc_(Pc), p_(p), eta_(eta) {
    ker_ = ker;
    const Moments rmom(Pr_, p_);
    const Moments cmom(Pc_, p_);
    rct_.init(rmom, 0, Pr_);
    cct_.init(cmom, 0, Pc_);
    hmat_.computePattern(rct_, cct_, eta);
  };

  FMCA::Matrix statistics() const { return hmat_.statistics(); }

  FMCA::iVector rindices() const {
    return Eigen::Map<const FMCA::iVector>(rct_.indices(), rct_.block_size());
  }
  FMCA::iVector cindices() const {
    return Eigen::Map<const FMCA::iVector>(cct_.indices(), cct_.block_size());
  }
  FMCA::Matrix action(const FMCA::Matrix &rhs) const {
    using Permutation =
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
    const Moments rmom(Pr_, p_);
    const Moments cmom(Pc_, p_);
    const usMatrixEvaluator mat_eval(rmom, cmom, ker_);
    const FMCA::Matrix srhs =
        Permutation(cindices().cast<int>()).transpose() * rhs;
    const FMCA::Matrix res = hmat_.action(mat_eval, srhs);
    return Permutation(rindices().cast<int>()) * res;
  }
  // member variables
  FMCA::CovarianceKernel ker_;
  H2Matrix hmat_;
  H2ClusterTree rct_;
  H2ClusterTree cct_;
  FMCA::Matrix Pr_;
  FMCA::Matrix Pc_;
  FMCA::Index p_;
  FMCA::Scalar eta_;
};
////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief class providing Samplet kernel approximations
 *
 **/
struct pySampletKernelCompressor {
  pySampletKernelCompressor() {}
  pySampletKernelCompressor(const pySampletTree &hst,
                            const FMCA::CovarianceKernel &ker,
                            const FMCA::Matrix &P, const FMCA::Scalar eta = 0.8,
                            const FMCA::Scalar thres = 0)
      : eta_(eta), thres_(thres), n_(P.cols()) {
    init(hst, ker, P, eta, thres);
  }

  template <typename Functor>
  FMCA::Vector matrixColumnGetter(const FMCA::Matrix &P,
                                  const FMCA::Index *idcs, const Functor &fun,
                                  FMCA::Index colID) {
    FMCA::Vector retval(P.cols());
    retval.setZero();
    for (auto i = 0; i < retval.size(); ++i)
      retval(i) = fun(P.col(idcs[i]), P.col(idcs[colID]));
    return retval;
  }

  void init(const pySampletTree &hst, const FMCA::CovarianceKernel &ker,
            const FMCA::Matrix &P, const FMCA::Scalar eta = 0.8,
            const FMCA::Scalar thres = 0) {
    const Moments mom(P, hst.p_);
    const MatrixEvaluator mat_eval(mom, ker);
    n_ = P.cols();
    eta_ = eta;
    thres_ = thres;
    std::cout << "mpole deg:                    " << hst.p_ << std::endl;
    std::cout << "dtilde:                       " << hst.dtilde_ << std::endl;
    std::cout << "eta:                          " << eta << std::endl;
    std::cout << "thres:                        " << thres << std::endl;
    {
      FMCA::internal::SampletMatrixCompressor<H2SampletTree> scomp;
      scomp.init(hst.ST_, eta, thres);
      scomp.compress(mat_eval);
      trips_ = scomp.triplets();
    }
    std::cout << "anz:                          "
              << std::round(trips_.size() / FMCA::Scalar(P.cols()))
              << std::endl;
    FMCA::Vector x(P.cols()), y1(P.cols()), y2(P.cols());
    FMCA::Scalar err = 0;
    FMCA::Scalar nrm = 0;
    for (auto i = 0; i < 10; ++i) {
      FMCA::Index index = rand() % P.cols();
      x.setZero();
      x(index) = 1;
      y1 = matrixColumnGetter(P, hst.ST_.indices(), ker, index);
      x = hst.ST_.sampletTransform(x);
      y2.setZero();
      for (const auto &i : trips_) {
        y2(i.row()) += i.value() * x(i.col());
        if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
      }
      y2 = hst.ST_.inverseSampletTransform(y2);
      err += (y1 - y2).squaredNorm();
      nrm += y1.squaredNorm();
    }
    err = sqrt(err / nrm);
    err_ = err;
    std::cout << "compression error:            " << err << std::endl;
  }

  Eigen::SparseMatrix<FMCA::Scalar> matrix() {
    Eigen::SparseMatrix<FMCA::Scalar> retval(n_, n_);
    retval.setFromTriplets(trips_.begin(), trips_.end());
    return retval;
  }

  FMCA::Index nnz() const { return trips_.size(); }
  FMCA::Scalar anz() const { return FMCA::Scalar(trips_.size()) / n_; }
  FMCA::Scalar error() const { return err_; }

  // member variables
  std::vector<Eigen::Triplet<FMCA::Scalar>> trips_;
  FMCA::Scalar eta_;
  FMCA::Scalar thres_;
  FMCA::Scalar err_;
  FMCA::Index n_;
};
////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief class providing Cholesky kernel approximations
 *
 **/
////////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(FMCA, m) {
  m.doc() = "pybind11 FMCA plugin";  // optional module docstring
  //////////////////////////////////////////////////////////////////////////////
  // ClusterTree
  //////////////////////////////////////////////////////////////////////////////
  py::class_<FMCA::ClusterTree> ClusterTree_(m, "ClusterTree");
  ClusterTree_.def(py::init<>());
  ClusterTree_.def(py::init<const FMCA::Matrix &, FMCA::Index>());
  m.def(
      "clusterTreeStatistics",
      [](const FMCA::ClusterTree &tree, const FMCA::Matrix &P) {
        return FMCA::clusterTreeStatistics(tree, P);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Displays metrics of a cluster tree");
  m.def(
      "kNN",
      [](const FMCA::ClusterTree &tree, const FMCA::Matrix &P, FMCA::Index k) {
        return FMCA::kNN(tree, P, k);
      },
      py::arg().noconvert(), py::arg().noconvert(), py::arg(),
      "return the list of the k-nearest neighbours of the points in P");
  //////////////////////////////////////////////////////////////////////////////
  // SampletTree
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pySampletTree> pySampletTree_(m, "SampletTree");
  pySampletTree_.def(py::init<>());
  pySampletTree_.def(py::init<const FMCA::Matrix &, FMCA::Index>());
  pySampletTree_.def("indices", &pySampletTree::indices);
  pySampletTree_.def("toNaturalOrder", &pySampletTree::toNaturalOrder);
  pySampletTree_.def("toClusterOrder", &pySampletTree::toClusterOrder);
  pySampletTree_.def("levels", &pySampletTree::levels);
  pySampletTree_.def("adaptiveTreeLeafPartition",
                     &pySampletTree::adaptiveTreeLeafPartition, py::arg("data"),
                     py::arg("thres"),
                     "labels each point by the leaf of the adaptive tree it "
                     "belongs to. thres is relative to the data energy");
  pySampletTree_.def("adaptiveTreeLeafBoxes",
                     &pySampletTree::adaptiveTreeLeafBoxes, py::arg("data"),
                     py::arg("thres"),
                     "bounding boxes of the leaves of the adaptive tree");
  pySampletTree_.def("levelLabels", &pySampletTree::levelLabels, py::arg("lvl"),
                     "labels each point by the cluster on level lvl it "
                     "belongs to");
  pySampletTree_.def("levelBoxes", &pySampletTree::levelBoxes, py::arg("lvl"),
                     "bounding boxes of the clusters on level lvl");
  pySampletTree_.def("sampletTransform", &pySampletTree::sampletTransform,
                     py::arg().noconvert(),
                     "samplet transform of cluster ordered data");
  pySampletTree_.def("inverseSampletTransform",
                     &pySampletTree::inverseSampletTransform,
                     py::arg().noconvert(),
                     "inverse samplet transform of samplet coefficients");
  pySampletTree_.def("sampletTransformMatrix",
                     &pySampletTree::sampletTransformMatrix,
                     py::arg().noconvert(),
                     "two sided samplet transform T * M * T^T of a cluster "
                     "ordered matrix M");
  pySampletTree_.def("inverseSampletTransformMatrix",
                     &pySampletTree::inverseSampletTransformMatrix,
                     py::arg().noconvert(),
                     "two sided inverse samplet transform T^T * M * T");
  pySampletTree_.def("transformationMatrix",
                     &pySampletTree::transformationMatrix,
                     "the samplet transform as a sparse matrix T");
  pySampletTree_.def("dim", &pySampletTree::dim);
  pySampletTree_.def("npts", &pySampletTree::npts);
  pySampletTree_.def("dtilde", &pySampletTree::dtilde);
  pySampletTree_.def("nscalfs", &pySampletTree::nscalfs,
      "number of leading scaling function coefficients");
  pySampletTree_.def("nclusters", &pySampletTree::nclusters);
  pySampletTree_.def("coeff2indices", &pySampletTree::coeff2indices);
  py::class_<pySampletTreeRP> pySampletTreeRP_(m, "SampletTreeRP");
  pySampletTreeRP_.def(py::init<>());
  pySampletTreeRP_.def(
      py::init<const FMCA::Matrix &, FMCA::Index, FMCA::Index>(), py::arg("P"),
      py::arg("dtilde"), py::arg("seed") = 0);
  pySampletTreeRP_.def("indices", &pySampletTreeRP::indices);
  pySampletTreeRP_.def("levels", &pySampletTreeRP::levels);
  pySampletTreeRP_.def("toNaturalOrder", &pySampletTreeRP::toNaturalOrder);
  pySampletTreeRP_.def("toClusterOrder", &pySampletTreeRP::toClusterOrder);
  pySampletTreeRP_.def("adaptiveTreeLeafPartition",
                       &pySampletTreeRP::adaptiveTreeLeafPartition,
                       py::arg("data"), py::arg("thres"),
                       "labels each point by the leaf of the adaptive tree it "
                       "belongs to. thres is relative to the data energy");
  pySampletTreeRP_.def("adaptiveTreeLeafBoxes",
                       &pySampletTreeRP::adaptiveTreeLeafBoxes,
                       py::arg("data"), py::arg("thres"),
                       "bounding boxes of the leaves of the adaptive tree");
  pySampletTreeRP_.def("levelLabels", &pySampletTreeRP::levelLabels,
                       py::arg("lvl"),
                       "labels each point by the cluster on level lvl it "
                       "belongs to");
  pySampletTreeRP_.def("levelBoxes", &pySampletTreeRP::levelBoxes,
                       py::arg("lvl"),
                       "bounding boxes of the clusters on level lvl");
  pySampletTreeRP_.def("sampletTransform", &pySampletTreeRP::sampletTransform,
                       py::arg().noconvert(),
                       "samplet transform of cluster ordered data");
  pySampletTreeRP_.def("inverseSampletTransform",
                       &pySampletTreeRP::inverseSampletTransform,
                       py::arg().noconvert(),
                       "inverse samplet transform of samplet coefficients");
  pySampletTreeRP_.def("sampletTransformMatrix",
                       &pySampletTreeRP::sampletTransformMatrix,
                       py::arg().noconvert(),
                       "two sided samplet transform T * M * T^T of a cluster "
                       "ordered matrix M");
  pySampletTreeRP_.def("inverseSampletTransformMatrix",
                       &pySampletTreeRP::inverseSampletTransformMatrix,
                       py::arg().noconvert(),
                       "two sided inverse samplet transform T^T * M * T");
  pySampletTreeRP_.def("transformationMatrix",
                       &pySampletTreeRP::transformationMatrix,
                       "the samplet transform as a sparse matrix T");
  pySampletTreeRP_.def("dim", &pySampletTreeRP::dim);
  pySampletTreeRP_.def("npts", &pySampletTreeRP::npts);
  pySampletTreeRP_.def("dtilde", &pySampletTreeRP::dtilde);
  pySampletTreeRP_.def("nscalfs", &pySampletTreeRP::nscalfs,
      "number of leading scaling function coefficients");
  pySampletTreeRP_.def("nclusters", &pySampletTreeRP::nclusters);
  pySampletTreeRP_.def("coeff2indices", &pySampletTreeRP::coeff2indices);
  m.def(
      "sampletTreeStatistics",
      [](const pySampletTree &tree, const FMCA::Matrix &P) {
        return FMCA::clusterTreeStatistics(tree.ST_, P);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Displays metrics of a samplet tree");
  m.def(
      "sampletTreeStatistics",
      [](const pySampletTreeRP &tree, const FMCA::Matrix &P) {
        return FMCA::clusterTreeStatistics(tree.ST_, P);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Displays metrics of a samplet tree");
  m.def(
      "sampletTransform",
      [](const pySampletTree &tree, const FMCA::Matrix &data) {
        return tree.ST_.sampletTransform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Performs samplet transform of data");
  m.def(
      "sampletTransform",
      [](const pySampletTreeRP &tree, const FMCA::Matrix &data) {
        return tree.ST_.sampletTransform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Performs samplet transform of data");
  m.def(
      "inverseSampletTransform",
      [](const pySampletTree &tree, const FMCA::Matrix &data) {
        return tree.ST_.inverseSampletTransform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Performs inverse samplet transform of data");
  m.def(
      "inverseSampletTransform",
      [](const pySampletTreeRP &tree, const FMCA::Matrix &data) {
        return tree.ST_.inverseSampletTransform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Performs inverse samplet transform of data");

  m.def(
      "sampletTransformMinLevel",
      [](const pySampletTree &tree, const FMCA::Matrix &data,
         const FMCA::Index min_level) {
        FMCA::internal::SampletTransformer<H2SampletTree> s_trafo(tree.ST_,
                                                                  min_level);
        return s_trafo.transform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
      "Performs samplet transform of data");
  m.def(
      "kNN",
      [](const pySampletTree &tree, const FMCA::Matrix &P, FMCA::Index k) {
        return FMCA::kNN(tree.ST_, P, k);
      },
      py::arg().noconvert(), py::arg().noconvert(), py::arg(),
      "return the list of the k-nearest neighbours of the points in P");

  //////////////////////////////////////////////////////////////////////////////
  // CovarianceKernel
  //////////////////////////////////////////////////////////////////////////////
  py::class_<FMCA::CovarianceKernel> pyCovarianceKernel_(m, "CovarianceKernel");
  pyCovarianceKernel_.def(py::init<>());
  pyCovarianceKernel_.def(py::init<const std::string &>());
  pyCovarianceKernel_.def(py::init<const std::string &, FMCA::Scalar>());
  pyCovarianceKernel_.def(
      py::init<const std::string &, FMCA::Scalar, FMCA::Scalar>());
  pyCovarianceKernel_.def(py::init<const std::string &, FMCA::Scalar,
                                   FMCA::Scalar, FMCA::Scalar>());
  pyCovarianceKernel_.def("kernelType", &FMCA::CovarianceKernel::kernelType);
  pyCovarianceKernel_.def("eval", &FMCA::CovarianceKernel::eval,
                          py::arg().noconvert(), py::arg().noconvert());
  //////////////////////////////////////////////////////////////////////////////
  // H2Matrix
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pyH2Matrix> pyH2Matrix_(m, "H2Matrix");
  pyH2Matrix_.def(
      py::init<const FMCA::CovarianceKernel &, const FMCA::Matrix &,
               const FMCA::Matrix &, const FMCA::Index, const FMCA::Scalar>());
  pyH2Matrix_.def("statistics", &pyH2Matrix::statistics);
  pyH2Matrix_.def("action", &pyH2Matrix::action, py::arg().noconvert(),
                  "computes the matrix-vector product");
  //////////////////////////////////////////////////////////////////////////////
  // SampletCompressor
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pySampletKernelCompressor> pySampletKernelCompressor_(
      m, "SampletKernelCompressor");
  pySampletKernelCompressor_.def(py::init<>());
  pySampletKernelCompressor_.def(
      py::init<const pySampletTree &, const FMCA::CovarianceKernel &,
               const FMCA::Matrix &, const FMCA::Scalar, const FMCA::Scalar>());
  pySampletKernelCompressor_.def("compute", &pySampletKernelCompressor::init,
                                 py::arg().noconvert(), py::arg().noconvert(),
                                 py::arg().noconvert(), py::arg(), py::arg(),
                                 "computes the compressed kernel");
  pySampletKernelCompressor_.def("matrix", &pySampletKernelCompressor::matrix,
                                 "returns the compressed kernel matrix, "
                                 "stored as upper triangular part");
  pySampletKernelCompressor_.def("nnz", &pySampletKernelCompressor::nnz,
                                 "number of stored matrix entries");
  pySampletKernelCompressor_.def("anz", &pySampletKernelCompressor::anz,
                                 "stored matrix entries per row");
  pySampletKernelCompressor_.def("error", &pySampletKernelCompressor::error,
                                 "estimated relative compression error");
  //////////////////////////////////////////////////////////////////////////////
  // pivoted Cholesky decomposition
  //////////////////////////////////////////////////////////////////////////////
  py::class_<FMCA::PivotedCholesky> pyPivotedCholesky_(m, "PivotedCholesky");
  pyPivotedCholesky_.def(py::init<>());
  pyPivotedCholesky_.def(py::init<const FMCA::CovarianceKernel &,
                                  const FMCA::Matrix &, FMCA::Scalar>());
  pyPivotedCholesky_.def("compute", &FMCA::PivotedCholesky::compute,
                         py::arg().noconvert(), py::arg().noconvert(),
                         py::arg(),
                         "Computes the pivoted Cholesky decomposition");
  pyPivotedCholesky_.def(
      "computeOMP", &FMCA::PivotedCholesky::computeOMP, py::arg().noconvert(),
      py::arg().noconvert(), py::arg().noconvert(), py::arg(),
      "Computes the pivoted Cholesky decomposition using OMP");
  pyPivotedCholesky_.def("computeBiorthogonalBasis",
                         &FMCA::PivotedCholesky::computeBiorthogonalBasis,
                         "Computes the biorthogonal basis");
  pyPivotedCholesky_.def("spectralBasisWeights",
                         &FMCA::PivotedCholesky::spectralBasisWeights,
                         "returns the transformation for the spectral basis");
  pyPivotedCholesky_.def(
      "computeFullPiv", &FMCA::PivotedCholesky::computeFullPiv,
      py::arg().noconvert(), py::arg().noconvert(), py::arg(),
      "Computes the truncated spectral decomposition");
  pyPivotedCholesky_.def("indices", &FMCA::PivotedCholesky::indices);
  pyPivotedCholesky_.def("matrixL", &FMCA::PivotedCholesky::matrixL);
  pyPivotedCholesky_.def("matrixU", &FMCA::PivotedCholesky::matrixU);
  pyPivotedCholesky_.def("matrixB", &FMCA::PivotedCholesky::matrixB);
  pyPivotedCholesky_.def("eigenvalues", &FMCA::PivotedCholesky::eigenvalues);
  //////////////////////////////////////////////////////////////////////////////
  // FALKON
  //////////////////////////////////////////////////////////////////////////////
  py::class_<FMCA::FALKON> pyFALKON_(m, "FALKON");
  pyFALKON_.def(py::init<>());
  pyFALKON_.def(py::init<const FMCA::CovarianceKernel &, const FMCA::Matrix &,
                         FMCA::Index, FMCA::Scalar>());
  pyFALKON_.def("init", &FMCA::FALKON::init, py::arg().noconvert(),
                py::arg().noconvert(), py::arg(), py::arg(),
                "Initializes FALKON by computing centers and matrices");
  pyFALKON_.def(
      "computeAlpha", &FMCA::FALKON::computeAlpha, py::arg().noconvert(),
      py::arg(),
      "computes coefficients for the initialized centers and the given RHS");
  pyFALKON_.def("indices", &FMCA::FALKON::indices);
  pyFALKON_.def("matrixC", &FMCA::FALKON::matrixC);
  pyFALKON_.def("matrixKPC", &FMCA::FALKON::matrixKPC);
}
