#ifndef FMCA_GRADKERNEL_GRADKERNEL_H_
#define FMCA_GRADKERNEL_GRADKERNEL_H_

namespace FMCA {
    class GradKernel {
    public:
        GradKernel(const FMCA::CovarianceKernel &kernel) : kernel_(kernel) {}

        std::vector<FMCA::Matrix> compute(const FMCA::Matrix &PR, const FMCA::Matrix &PC) const {
            std::vector<FMCA::Matrix> gradMatrices(3, FMCA::Matrix(PR.cols(), PC.cols()));

        // evaluate the Kernel
        for (int j = 0; j < PC.cols(); ++j) {
            for (int i = 0; i < PR.cols(); ++i) {
                auto kernelValue = kernel_.eval(PR.col(i), PC.col(j));

                //compute the gradient
                for (int d = 0; d < 3; ++d){
                    double diff = PR.col(i)[d] - PC.col(j)[d];
                    gradMatrices[d](i,j) = - diff * kernelValue.value();
                }
            }
        }
        return gradMatrices;
        }
    
        FMCA::Matrix compute_component(const FMCA::Matrix &PR, const FMCA::Matrix &PC, int d) const {
            if (d >= 0 && d < 3) {
                FMCA::Matrix gradMatrix(PR.cols(), PC.cols());

                // evaluate the Kernel and compute the gradient for dimension d
                for (int j = 0; j < PC.cols(); ++j) {
                    for (int i = 0; i < PR.cols(); ++i) {
                        auto kernelValue = kernel_.eval(PR.col(i), PC.col(j));
                        double diff = PR.col(i)[d] - PC.col(j)[d];
                        gradMatrix(i, j) = -diff * kernelValue.value();
                    }
                }

                return gradMatrix;
            } else {
                throw std::invalid_argument("Invalid dimension d. Must be 0, 1, or 2.");
            }
        }


    private:
        const FMCA::CovarianceKernel &kernel_;
    };

}  // namespace FMCA

#endif 

