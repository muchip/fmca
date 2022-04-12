#include <FMCA/Clustering>
#include <iostream>

#include "../../FMCA/src/util/IO.h"
#include "../../FMCA/src/util/Tictoc.h"
#include "../../FMCA/src/util/print2file.h"
#include "../../Points/matrixReader.h"
#include "generateSwissCheese.h"
#include "generateSwissCheeseExp.h"

template <typename Derived>
double computeDistance(const Derived &cluster1, const Derived &cluster2) {
  const double row_radius = 0.5 * cluster1.bb().col(2).norm();
  const double col_radius = 0.5 * cluster2.bb().col(2).norm();
  const double dist = 0.5 * (cluster1.bb().col(0) - cluster2.bb().col(0) +
                             cluster1.bb().col(1) - cluster2.bb().col(1))
                                .norm() -
                      row_radius - col_radius;
  return dist > 0 ? dist : 0;
}

template <typename Derived, typename Derived2>
double clusterSeparationRadius(double separation_radius,
                               const Derived &cluster1, const Derived &cluster2,
                               const Eigen::MatrixBase<Derived2> &P) {
  double retval = separation_radius;
  double rad = 0;
  double dist = computeDistance(cluster1, cluster2);

  if (separation_radius > 0.5 * dist) {
    if (cluster2.nSons()) {
      rad = 0.5 * (dist + cluster2.bb().col(2).norm());
      separation_radius = separation_radius < rad ? separation_radius : rad;
      for (auto i = 0; i < cluster2.nSons(); ++i) {
        rad = clusterSeparationRadius(separation_radius, cluster1,
                                      cluster2.sons(i), P);
        retval = retval < rad ? retval : rad;
      }
    } else {
      if (cluster1.block_id() != cluster2.block_id())
        for (auto j = 0; j < cluster1.indices().size(); ++j)
          for (auto i = 0; i < cluster2.indices().size(); ++i) {
            rad = 0.5 *
                  (P.col(cluster2.indices()[i]) - P.col(cluster1.indices()[j]))
                      .norm();
            retval = retval < rad ? retval : rad;
          }
    }
  }
  return retval;
}

template <typename Derived, typename Derived2>
void clusterFillDistance(Eigen::VectorXd &fill_distance,
                         const Derived &cluster1, const Derived &cluster2,
                         const Eigen::MatrixBase<Derived2> &P) {
  double max_dist = 0;
  double dist = computeDistance(cluster1, cluster2);
  // check cluster if there is the chance of improving the distance of a given
  // point
  if (fill_distance.maxCoeff() > dist) {
    if (cluster2.nSons()) {
      for (auto i = 0; i < cluster2.nSons(); ++i)
        clusterFillDistance(fill_distance, cluster1, cluster2.sons(i), P);
    } else {
      if (cluster1.block_id() != cluster2.block_id())
        for (auto j = 0; j < cluster1.indices().size(); ++j)
          for (auto i = 0; i < cluster2.indices().size(); ++i) {
            max_dist =
                (P.col(cluster2.indices()[i]) - P.col(cluster1.indices()[j]))
                    .norm();
            fill_distance(j) =
                fill_distance(j) > max_dist ? max_dist : fill_distance(j);
          }
    }
  }
  return;
}

int main() {
  FMCA::Tictoc T;
  // unsigned int npts = 1e5;`
  T.tic();
  Eigen::MatrixXd P;
  double separation_radius_full = double(1.) / double(0.);
  double fill_distance_full = 0;

#if 0
  =
      Eigen::MatrixXd::Random(3, 80000); // = generateSwissCheeseExp(2, npts);
  T.tic();
  for (auto j = 0; j < P.cols(); ++j) {
    double dist = double(1.) / double(0.);
    for (auto i = 0; i < P.cols(); ++i) {
      if (i != j) {
        double rad = 0.5 * (P.col(i) - P.col(j)).norm();
        dist = dist > (2 * rad) ? (2 * rad) : dist;
        separation_radius_full =
            separation_radius_full < rad ? separation_radius_full : rad;
      }
    }
    fill_distance_full = fill_distance_full < dist ? dist : fill_distance_full;
  }
  T.toc("time full rad: ");
  std::cout << "rad full distance matrix: " << separation_radius_full
            << std::endl;
  std::cout << "fill full distance matrix: " << fill_distance_full << std::endl;
#endif
  Eigen::MatrixXd Pts = readMatrix("../../Points/cross3D.txt");
  P = Pts.transpose();
  // Bembel::IO::print2m("d1c.m", "P", P, "w");
  // return 0;
  T.toc("pts... ");
  Eigen::MatrixXd Q;
  // Q.setZero();
  // Q.topRows(2) = P;
  Q = P;
  FMCA::ClusterTree CT(Q, 10);
  T.tic();
  double separation_radius = double(1.) / double(0.);
  for (auto &it : CT) {
    if (!it.nSons()) {
      double rad = 0;
      for (auto j = 0; j < it.indices().size(); ++j)
        for (auto i = j + 1; i < it.indices().size(); ++i) {
          rad = 0.5 * (P.col(it.indices()[i]) - P.col(it.indices()[j])).norm();
          separation_radius = separation_radius < rad ? separation_radius : rad;
        }
      rad = clusterSeparationRadius(separation_radius, it, CT, P);
      separation_radius = separation_radius < rad ? separation_radius : rad;
    }
  }
  T.toc("separation radius comp: ");
  std::cout << separation_radius << std::endl;
  std::cout << "err: " << std::abs(separation_radius - separation_radius_full)
            << std::endl;
  double fill_distance = double(0.);
  Eigen::VectorXd min_distance;
  double dist = 0;
  for (auto &it : CT) {
    if (!it.nSons()) {
      min_distance.resize(it.indices().size());
      min_distance.setOnes();
      min_distance /= double(0.);
      // determine candidate distances within the cluster
      for (auto j = 0; j < it.indices().size(); ++j)
        for (auto i = 0; i < it.indices().size(); ++i) {
          if (i != j) {
            dist = (P.col(it.indices()[i]) - P.col(it.indices()[j])).norm();
            min_distance(j) = min_distance(j) > dist ? dist : min_distance(j);
          }
        }
      clusterFillDistance(min_distance, it, CT, P);
      const double cluster_max_dist = min_distance.maxCoeff();
      fill_distance =
          fill_distance < cluster_max_dist ? cluster_max_dist : fill_distance;
    }
  }
  T.toc("fill distance comp: ");
  std::cout << fill_distance << std::endl;
  std::cout << "error: " << std::abs(fill_distance - fill_distance_full)
            << std::endl;

  T.tic();
  std::vector<const FMCA::TreeBase<FMCA::ClusterTree> *> leafs;
  for (auto level = 0; level < 16; ++level) {
    std::vector<Eigen::MatrixXd> bbvec;
    for (auto &node : CT) {
      if (node.level() == level) bbvec.push_back(node.derived().bb());
    }
    FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
  }
  std::vector<Eigen::MatrixXd> bbvec;

  for (auto &node : CT) {
    if (!node.nSons()) bbvec.push_back(node.derived().bb());
  }
  Eigen::VectorXd colrs(P.cols());
  for (auto &node : CT) {
    if (!node.nSons())
      for (auto it = node.indices().begin(); it != node.indices().end(); ++it)
        colrs(*it) = node.block_id();
  }
  FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
  FMCA::IO::plotPointsColor("points.vtk", Q, colrs);

  return 0;
}
