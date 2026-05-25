#include "ObjectiveFunctions.h"
#include <cmath>
#include <numbers> // C++20 for std::numbers::pi, or use M_PI

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MathFunctions {

    double DeJong1(const std::vector<double>& x) {
        double sum = 0.0;
        for (double val : x) {
            sum += val * val;
        }
        return sum;
    }

    double Schwefel(const std::vector<double>& x) {
        double sum = 0.0;
        for (double val : x) {
            sum += -val * std::sin(std::sqrt(std::abs(val)));
        }
        return sum;
    }

    double Rastrigin(const std::vector<double>& x) {
        double sum = 0.0;
        for (double val : x) {
            sum += (val * val - 10.0 * std::cos(2.0 * M_PI * val));
        }
        return 10.0 * x.size() + sum;
    }

    double Michalewicz(const std::vector<double>& x) {
        double sum = 0.0;
        double m = 10.0; // Steepness parameter
        for (size_t i = 0; i < x.size(); ++i) {
            double term = std::sin(x[i]) * std::pow(std::sin(((i + 1) * x[i] * x[i]) / M_PI), 2 * m);
            sum += term;
        }
        return -sum; // Minimization
    }

    void getBounds(const std::string& funcName, double& lower, double& upper) {
        if (funcName == "DeJong1") {
            lower = -5.12; upper = 5.12;
        } else if (funcName == "Schwefel") {
            lower = -500.0; upper = 500.0;
        } else if (funcName == "Rastrigin") {
            lower = -5.12; upper = 5.12;
        } else if (funcName == "Michalewicz") {
            lower = 0.0; upper = M_PI;
        } else {
            // Default fallback
            lower = -10.0; upper = 10.0;
        }
    }

    double getGlobalOptimum(const std::string& funcName, int dimension) {
        if (funcName == "DeJong1") return 0.0;
        if (funcName == "Schwefel") return -418.9829 * dimension;
        if (funcName == "Rastrigin") return 0.0;
        if (funcName == "Michalewicz") {
            // Approximate known optima for Michalewicz
            if (dimension == 5) return -4.687;
            if (dimension == 10) return -9.66;
            if (dimension == 30) return -29.6309; // Approx
        }
        return 0.0;
    }
}
