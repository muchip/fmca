#ifndef FMCA_MATRIXEVALUATORS_SPARSEEVALUATOR_H_
#define FMCA_MATRIXEVALUATORS_SPARSEEVALUATOR_H_

#include <Eigen/Sparse>
#include <vector>

namespace FMCA {

struct SparseEvaluator {
 public:
  using SparseMatrix = Eigen::SparseMatrix<double>;
  using Index = Eigen::Index;
  using Triplet = Eigen::Triplet<double>;

  // Constructor
  SparseEvaluator(Index rows, Index cols, const std::vector<Triplet>& triplets) {
    matrix_ = buildMatrix(rows, cols, triplets);
  }

  // Accessor for the matrix
  const SparseMatrix& matrix() const { return matrix_; }

 private:
  SparseMatrix matrix_;

  // Builds the sparse matrix
  SparseMatrix buildMatrix(Index rows, Index cols, const std::vector<Triplet>& triplets) {
    SparseMatrix mat(rows, cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
  }
};

}  // namespace FMCA
#endif

