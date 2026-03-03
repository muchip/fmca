#ifndef FMCA_KERNELINTERPOLATION_MULTISCALEINTERPOLATOR_H_
#define FMCA_KERNELINTERPOLATION_MULTISCALEINTERPOLATOR_H_

namespace FMCA {

template <typename KernelSolver>
class MultiscaleInterpolator {
 public:
  using Solver = KernelSolver;

  MultiscaleInterpolator() = default;  // ask Michael

  void init(const std::vector<Matrix>& P_levels, Index dtilde, Scalar eta = 0.,
            Scalar threshold = 0., Scalar ridgep = 0., Scalar nu = 1.0,
            Index dim = 2) {
    num_levels_ = P_levels.size();
    P_levels_ = P_levels;
    dtilde_ = dtilde;
    eta_ = eta;
    threshold_ = threshold;
    ridgep_ = ridgep;
    nu_ = nu;
    dim_ = dim;

    // initialize the solvers for each level
    solvers_.clear();
    solvers_.reserve(num_levels_);
    fill_distances_.clear();
    fill_distances_.reserve(num_levels_);

    for (Index l = 0; l < num_levels_; ++l) {
      solvers_.emplace_back();
      // Crea un oggetto vuoto direttamente nel vector. Non posso fare
      // solvers_.push_back(solver) perchè il copy constructor è = delete.
      // Chiedi a Michael.
      solvers_[l].init(P_levels_[l], dtilde_, eta_, threshold_, ridgep_);
      fill_distances_.push_back(solvers_[l].fill_distance());
    }

    return;
  }

  ///////////////////////////////////////////////////////////////////// getters
  // Solver at a specific level
  Solver& solver(Index level) { return solvers_[level]; }
  // Point set at a specific level
  const Matrix& points(Index level) const { return P_levels_[level]; }
  // Fill distance at a specific level
  Scalar fillDistance(Index level) const { return fill_distances_[level]; }
  // Number of levels
  Index numLevels() const { return num_levels_; }

 private:
  std::vector<Solver> solvers_;         // Vector of solvers, one per level
  std::vector<Matrix> P_levels_;        // Points at each level
  std::vector<Scalar> fill_distances_;  // Fill distance at each level
  Index num_levels_;
  Index dtilde_;
  Scalar eta_;
  Scalar threshold_;
  Scalar ridgep_;
  Scalar nu_;
  Index dim_;
};

}  // namespace FMCA

#endif  // FMCA_KERNELINTERPOLATION_MULTISCALEINTERPOLATOR_H_