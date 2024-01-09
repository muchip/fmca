#ifndef FMCA_GRADKERNEL_GRADKERNEL_H_
#define FMCA_GRADKERNEL_GRADKERNEL_H_

namespace FMCA {
    class GradKernel {
    public:
    GradKernel(){};
    GradKernel(const GradKernel &other) {
        gradkernel_ = other.gradkernel_;
        ktype_ = other.ktype_;
        l_ = other.l_;
        c_ = other.c_;
        d_ = other.d_;
    }

    GradKernel(GradKernel &&other) {
        gradkernel_ = other.gradkernel_;
        ktype_ = other.ktype_;
        l_ = other.l_;
        c_ = other.c_;
        d_ = other.d_;
    }

    GradKernel &operator=(GradKernel other) {
        std::swap(gradkernel_, other.gradkernel_);
        std::swap(ktype_, other.ktype_);
        std::swap(l_, other.l_);
        std::swap(c_, other.c_);
        std::swap(d_, other.d_);
        return *this;
    }

    GradKernel(const std::string &ktype, FMCA::Scalar l = 1., FMCA::Scalar c = 1., int d = 0)
        : ktype_(ktype), l_(l), c_(c), d_(d) {
        // Transform string to upper case
        for (auto &chr : ktype_) chr = (char)toupper(chr);
        ////////////////////////////////////////////////////////////////////////////
        if (ktype_ == "GAUSSIAN")
            gradkernel_ = [this](FMCA::Scalar x, FMCA::Scalar y, FMCA::Scalar r) {
                return - (x - y) / (l_ * l_) * exp(-0.5 * r * r / (l_ * l_));
            };
        else if (ktype_ == "MATERN32")
            gradkernel_ = [this](FMCA::Scalar x, FMCA::Scalar y, FMCA::Scalar r) {
                return - 3 * (x - y) / (l_ * l_) * exp(-sqrt(3) / l_ * r );
            };
        ////////////////////////////////////////////////////////////////////////////
        else if (ktype_ == "EXPONENTIAL")
            gradkernel_ = [this](FMCA::Scalar x, FMCA::Scalar y, FMCA::Scalar r) {
                return r < FMCA_ZERO_TOLERANCE ? std::numeric_limits<double>::quiet_NaN() :  - (x - y) / (r * l_) * exp(-r / l_);
            };
        ////////////////////////////////////////////////////////////////////////////
        else
            assert(false && "desired gradient kernel not implemented");
        }

    template <typename derived, typename otherDerived>

    FMCA::Scalar operator()(const Eigen::MatrixBase<derived>& x,const Eigen::MatrixBase<otherDerived>& y) const {
        return gradkernel_(x[d_], y[d_], (x - y).norm());
    }
    FMCA::Matrix eval(const FMCA::Matrix &PR, const FMCA::Matrix &PC) const {
        FMCA::Matrix retval(PR.cols(), PC.cols());
        for (auto j = 0; j < PC.cols(); ++j)
            for (auto i = 0; i < PR.cols(); ++i)
                retval(i, j) = operator()(PR.col(i), PC.col(j));
        return retval;
    }
    std::string gradkernelType() const { return ktype_; }

    private:
        std::function<FMCA::Scalar(FMCA::Scalar, FMCA::Scalar, FMCA::Scalar)> gradkernel_;
        std::string ktype_;
        FMCA::Scalar l_;
        FMCA::Scalar c_;
        int d_;
    };

}  // namespace FMCA

#endif
