#include "../FMCA/src/util/Tictoc.h"
#include <Eigen/Dense>
#include <falconn/lsh_nn_table.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

using falconn::compute_number_of_hash_functions;
using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::StorageHashTable;

int main() {
  using Point = DenseVector<float>;

  const int N = 70000; // number of points (columns)
  const int D = 784;   // dimension (rows)
  const int num_threads = 14;
  const int L = 10; // number of hash tables
  const int num_hash_bits = 18;

  FMCA::Tictoc T;

  Eigen::MatrixXf mat(D, N); // D rows, N cols
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  for (int i = 0; i < D; ++i)
    for (int j = 0; j < N; ++j)
      mat(i, j) = dist(rng);

  std::vector<Point> data;
  data.reserve(N);
  for (int j = 0; j < N; ++j) {
    data.emplace_back(Point(D));
    Eigen::Map<Eigen::VectorXf>(data.back().data(), D) = mat.col(j);
  }

  LSHConstructionParameters params;
  params.dimension = D;
  params.lsh_family = LSHFamily::CrossPolytope;
  params.l = L;
  params.distance_function = DistanceFunction::EuclideanSquared;
  params.num_rotations = 1; // dense data
  compute_number_of_hash_functions<Point>(num_hash_bits, &params);
  params.num_setup_threads = num_threads;
  params.storage_hash_table = StorageHashTable::FlatHashTable;

  T.tic();
  auto table = construct_table<Point>(data, params);

  auto query_pool = table->construct_query_pool(L, -1, num_threads);
  T.toc("init");

  std::vector<Point> queries(N);
  for (int j = 0; j < N; ++j) {
    queries[j] = Point(D);
    Eigen::Map<Eigen::VectorXf>(queries[j].data(), D) = mat.col(j);
  }

  std::vector<std::vector<int32_t>> all_neighbors(N);
#pragma omp parallel for
  for (int j = 0; j < N; ++j) {
    query_pool->find_near_neighbors(queries[j], 1.5, &all_neighbors[j]);
  }

  T.toc("total");

  return 0;
}

// #include "../FMCA/src/util/Tictoc.h"
// #include <Eigen/Dense>
// #include <falconn/lsh_nn_table.h>
// #include <iostream>
// #include <random>
// #include <vector>

// using falconn::compute_number_of_hash_functions;
// using falconn::construct_table;
// using falconn::DenseVector;
// using falconn::DistanceFunction;
// using falconn::LSHConstructionParameters;
// using falconn::LSHFamily;
// using falconn::StorageHashTable;

// using Point = DenseVector<float>;

// int main() {
//   const int N = 70000;
//   const int D = 784;
//   int num_hash_bits = 18; // log2(N)
//   int num_threads = 14;   // omp_get_max_threads()
//   FMCA::Tictoc T;
//   Eigen::MatrixXf mat(N, D);
//   std::mt19937 rng(0);
//   std::uniform_real_distribution<float> dist(0.0, 1.0);
//   for (int i = 0; i < N; ++i)
//     for (int j = 0; j < D; ++j)
//       mat(i, j) = dist(rng);

//   // Convert to std::vector<Point>
//   std::vector<Point> data;
//   data.reserve(N);
//   for (int i = 0; i < N; ++i) {
//     Point p(D);
//     for (int j = 0; j < D; ++j)
//       p[j] = mat(i, j);
//     data.push_back(std::move(p));
//   }

//   T.tic();
//   // LSH parameters
//   int L = 10;
//   LSHConstructionParameters params;
//   params.dimension = D;
//   params.lsh_family = LSHFamily::CrossPolytope;
//   params.l = L; //
//   params.distance_function = DistanceFunction::EuclideanSquared;
//   params.num_rotations = 1; // dense data
//   compute_number_of_hash_functions<Point>(18, &params);
//   params.num_setup_threads = num_threads;
//   params.storage_hash_table = StorageHashTable::FlatHashTable;

//   auto table = construct_table<Point>(data, params);

//   // Build FALCONN query object
//   auto query_pool = table->construct_query_pool(L, -1, num_threads);

//   std::vector<Point> queries(N);
//   for (int i = 0; i < N; ++i) {
//     queries[i] = Point(D);
//     for (int j = 0; j < D; ++j)
//       queries[i][j] = mat(i, j);
//   }

//   std::vector<std::vector<int32_t>> all_neighbors(N);

// // Parallel batch query
// #pragma omp parallel for
//   for (int i = 0; i < N; ++i) {
//     query_pool->find_near_neighbors(queries[i], 1.5, &all_neighbors[i]);
//   }

//   // Use last row of mat as query vector

//   Point query_vector(D);

//   for (int i = 0; i < N; i++) {
//     for (int j = 0; j < D; ++j)
//       query_vector[j] = mat(i, j);
//     std::vector<int32_t> neighbors;
//     query_object->find_near_neighbors(query_vector, 1.5, &neighbors);
//   }

//   T.toc("toc end");
// }
