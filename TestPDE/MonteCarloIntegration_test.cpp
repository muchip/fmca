#include <iostream>
#include <vector>
#include <functional>
#include <stdexcept>
#include <MonteCarloIntegration.h>

double testFunction(const std::vector<double>& point) {
    return point[0] * point[0] + point[1] * point[1];
}

bool domainCheck(const std::vector<double>& point) {
    return (point[0] >= 0.0 && point[0] <= 1.0 && point[1] >= 0.0 && point[1] <= 1.0);
}

int main() {
        std::vector<double> minLimits = {0.0, 0.0};
        std::vector<double> maxLimits = {1.0, 1.0};
        int samples = 100000;
        double result = monteCarloIntegration(testFunction, domainCheck, minLimits, maxLimits, samples);
        std::cout << "Monte Carlo integration result: " << result << std::endl;
    return 0;
}
