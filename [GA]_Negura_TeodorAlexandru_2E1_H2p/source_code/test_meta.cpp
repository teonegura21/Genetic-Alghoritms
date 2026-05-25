#include <iostream>
#include "MetaGA.h"
#include "ObjectiveFunctions.h"

// Simple test to verify Meta-GA compiles and optimizes one configuration
int main() {
    std::cout << "=== Meta-GA Quick Test ===" << std::endl;
    std::cout << "Testing: Rastrigin 5D" << std::endl << std::endl;
    
    // Create a simple test - optimize Rastrigin 5D
    MetaGA metaGA("Rastrigin", 5, MathFunctions::Rastrigin, 
                 10, // Small meta-population for quick test
                 5,  // Few generations 
                 3,  // 3 trials per config
                 500); // Short base GA runs
    
    GAConfig bestConfig = metaGA.optimize();
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "Best Configuration:" << std::endl;
    std::cout << "  Pop Size: " << bestConfig.populationSize << std::endl;
    std::cout << "  Mut Rate: " << bestConfig.mutationRate << std::endl;
    std::cout << "  Cross Rate: " << bestConfig.crossoverRate << std::endl;
    std::cout << "  Eta_c: " << bestConfig.eta_c << std::endl;
    std::cout << "  Eta_m: " << bestConfig.eta_m << std::endl;
    std::cout << "  Fitness: " << bestConfig.fitness << std::endl;
    
    return 0;
}
