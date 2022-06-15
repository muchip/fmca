// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
#define FMCA_CLUSTERSET_
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/Sparse>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/SparseMatrix.h>
#include <FMCA/src/util/print2file.h>

#include "pardiso_interface.h"
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  Eigen::MatrixXd mtrips;
  FMCA::IO::bin2Mat("matrix.dat", &mtrips);
  const unsigned int npts = mtrips(0, 0);
  const unsigned int n_triplets = mtrips.rows() - 1;
  std::cout << "n: " << npts << " nnz: " << n_triplets << std::endl;
  for (auto i = 1; i < mtrips.size(); ++i)
    if (mtrips(i + 1, 0) == mtrips(i, 0) && mtrips(i + 1, 1) == mtrips(i, 1))
      assert(false && "duplicate entry in input");
  unsigned int n_diag_elems = 0;
  std::vector<Eigen::Triplet<double>> trips(n_triplets);
  for (auto i = 1; i < mtrips.rows(); ++i) {
    trips[i - 1] = Eigen::Triplet<double>(int(mtrips(i, 0)), int(mtrips(i, 1)),
                                          mtrips(i, 2));
    if (mtrips(i, 0) == mtrips(i, 1)) ++n_diag_elems;
  }
  for (auto i = 1; i < trips.size(); ++i)
    if (trips[i].row() == trips[i - 1].row() &&
        trips[i].col() == trips[i - 1].col())
      assert(false && "duplicate entry in input triplets");
  mtrips.resize(0, 0);
  std::cout << "number of diagonal elements: " << n_diag_elems << std::endl;
  FMCA::SparseMatrix<double> S(npts, npts);
  std::cout << "writing into sparse matrix" << std::endl;
  S.setFromTriplets(trips.begin(), trips.end());
  trips.clear();
  trips = S.toTriplets();
  FMCA::SparseMatrix<double>::sortTripletsInPlace(trips);
  std::cout << "returned triplets" << std::endl;
  std::cout << trips.size() << std::endl;
  for (auto i = 1; i < trips.size(); ++i)
    if (trips[i].row() == trips[i - 1].row() &&
        trips[i].col() == trips[i - 1].col())
      assert(false && "duplicate entry");
  int i = 0;
  int j = 0;
  int n = npts;
  std::cout << "triplet size (\% of INT_MAX):"
            << (long double)(n_triplets) / (long double)(INT_MAX)*100
            << std::endl;
  assert(n_triplets < INT_MAX && "exceeded INT_MAX");
  int *ia = nullptr;
  int *ja = nullptr;
  double *a = nullptr;
  ia = (int *)malloc((n + 1) * sizeof(int));
  ja = (int *)malloc(n_triplets * sizeof(int));
  a = (double *)malloc(n_triplets * sizeof(double));
  memset(ia, 0, (n + 1) * sizeof(int));
  memset(ja, 0, n_triplets * sizeof(int));
  memset(a, 0, n_triplets * sizeof(double));
  // write rows
  ia[trips[0].row()] = 0;
  for (i = trips[0].row() + 1; i <= n; ++i) {
    while (j < n_triplets && i - 1 == trips[j].row()) ++j;
    ia[i] = j;
  }
  assert(j == n_triplets && "j is not ntriplets");
  // write the rest
  for (i = 0; i < n_triplets; ++i) {
    ja[i] = trips[i].col();
    a[i] = trips[i].value();
  }
  std::cout << "\n\nentering pardiso block\n" << std::flush;
  std::printf("ia=%p ja=%p a=%p n=%i nnz=%i\n", ia, ja, a, n, ia[n]);
  std::cout << std::flush;
  pardiso_interface(ia, ja, a, n);

  free(ia);
  free(ja);
  free(a);
  return 0;
}
