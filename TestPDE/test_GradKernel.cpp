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

    // evaluate the Gaussian Kernel gradient
    for (int d = 0; d < 3; ++d){
        FMCA::GradKernel gradkernel("GAUSSIAN", 1, 1, d);
        auto gradkernelValue = gradkernel.eval(PR, PC);
        std::cout << "Gradient Matrix " << d + 1 << ":\n" << gradkernelValue << "\n\n";
    }

    return 0;
}
