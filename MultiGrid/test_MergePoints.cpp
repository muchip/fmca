#include <iostream>
#include <vector>
#include <set>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Point = std::pair<double, double>;

// Function to merge matrices ensuring only unique points are added
void mergeMatrices(const MatrixXd& src, std::vector<Point>& dest) {
    std::set<Point> uniquePoints(dest.begin(), dest.end());
    for (int i = 0; i < src.cols(); ++i) {
        Point point(src(0, i), src(1, i));
        if (uniquePoints.insert(point).second) {
            dest.push_back(point);
        }
    }
}

int main() {
    int DIM = 2; // Assuming 2D points

    // Define example matrices
    MatrixXd X1(DIM, 3);
    X1 << 1, 2, 3,
          1, 2, 3;

    MatrixXd X2(DIM, 4);
    X2 << 1, 10, 2, 3,
          1, 10, 2, 3;

    MatrixXd X3(DIM, 6);
    X3 << 1, 6, 2, 11, 3, 10,
          1, 6, 2, 11, 3, 10;

    // Create the final vector of unique points
    std::vector<Point> uniquePoints;

    // Merge matrices into the vector of unique points
    mergeMatrices(X1, uniquePoints);
    mergeMatrices(X2, uniquePoints);
    mergeMatrices(X3, uniquePoints);

    // Convert the vector of unique points back to an Eigen::Matrix
    MatrixXd X(DIM, uniquePoints.size());
    for (size_t i = 0; i < uniquePoints.size(); ++i) {
        X(0, i) = uniquePoints[i].first;
        X(1, i) = uniquePoints[i].second;
    }

    // Print the resulting matrix X
    std::cout << "Resulting matrix X:\n" << X << std::endl;

    return 0;
}
