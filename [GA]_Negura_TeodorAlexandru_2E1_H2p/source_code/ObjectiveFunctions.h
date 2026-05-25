#ifndef OBJECTIVE_FUNCTIONS_H
#define OBJECTIVE_FUNCTIONS_H

#include <vector>
#include <cmath>
#include <string>

// Namespace to encapsulate all mathematical objective functions.
// This promotes modularity and avoids global namespace pollution.
namespace MathFunctions {

    // De Jong 1 (Sphere) Function
    // Global Minimum: 0 at x = [0, 0, ..., 0]
    // Search Domain: [-5.12, 5.12] usually
    double DeJong1(const std::vector<double>& x);

    // Schwefel's Function
    // Global Minimum: 0 at x = [420.9687, ..., 420.9687]
    // Search Domain: [-500, 500]
    double Schwefel(const std::vector<double>& x);

    // Rastrigin's Function
    // Global Minimum: 0 at x = [0, 0, ..., 0]
    // Search Domain: [-5.12, 5.12]
    double Rastrigin(const std::vector<double>& x);

    // Michalewicz's Function
    // Global Minimum: Depends on dimension (e.g., -4.687 for d=5)
    // Search Domain: [0, PI]
    double Michalewicz(const std::vector<double>& x);

    // Helper to get bounds for a specific function name
    void getBounds(const std::string& funcName, double& lower, double& upper);

    // Helper to get the known global optimum for a function and dimension
    // Used to calculate the "Delta" (Error)
    double getGlobalOptimum(const std::string& funcName, int dimension);
}

#endif // OBJECTIVE_FUNCTIONS_H
