// thi test is not compiled, it is an old verision of the test_GradKernel.cpp that uses GradKernel1.h

#include <iostream>
#include <Eigen/Dense>
#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"

int main() {
    FMCA::Matrix PR(3, 3);
    PR << 1, 1, 3,
          1, 2, 2,
          1, 2, 1;
    std::cout << PR << std::endl;
    FMCA::Matrix PC(3, 3);
    PC << 0, 0, 3,
          0, 1, 1,
          0, 0, 1;

    FMCA::CovarianceKernel kernel("GAUSSIAN", 1);
    for (int j = 0; j < PC.cols(); ++j) {
    for (int i = 0; i < PR.cols(); ++i) {
        auto kernelValue = kernel.eval(PR.col(i), PC.col(j));
        std::cout << "i = " << i << std::endl;
        std::cout << "j = " << j << std::endl;
        std::cout << "K(i,j) = " << kernelValue << std::endl;
    }
    }


    // Create GradKernel instance
    FMCA::GradKernel1 gradKernel(kernel);

    // Compute gradients
    std::vector<FMCA::Matrix> gradMatrices = gradKernel.compute(PR, PC);

    // Print the results (optional, you can modify this based on your needs)
    for (int d = 0; d < 3; ++d) {
        std::cout << "Gradient Matrix " << d + 1 << ":\n" << gradMatrices[d] << "\n\n";
    }

        return 0;
}
