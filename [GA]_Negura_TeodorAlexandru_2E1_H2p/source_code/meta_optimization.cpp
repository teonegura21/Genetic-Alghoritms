#include <iostream>
#include <string>
#include <map>
#include <filesystem>
#include "MetaGA.h"
#include "ObjectiveFunctions.h"

int main() {
    std::cout << "=== Meta-Genetic Algorithm: Hyperparameter Optimization ===" << std::endl;
    std::cout << "Optimizing GA parameters for all 12 configurations." << std::endl;
    std::cout << "This will take some time..." << std::endl << std::endl;
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories("Results/optimized_params");
    
    std::vector<std::string> functions = {"DeJong1", "Schwefel", "Rastrigin", "Michalewicz"};
    std::vector<int> dimensions = {5, 10, 30};
    
    // Map function names to function pointers
    std::map<std::string, std::function<double(const std::vector<double>&)>> funcMap = {
        {"DeJong1", MathFunctions::DeJong1},
        {"Schwefel", MathFunctions::Schwefel},
        {"Rastrigin", MathFunctions::Rastrigin},
        {"Michalewicz", MathFunctions::Michalewicz}
    };
    
    int totalConfigs = functions.size() * dimensions.size();
    int currentConfig = 0;
    
    for (const auto& funcName : functions) {
        for (int dim : dimensions) {
            currentConfig++;
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "Configuration " << currentConfig << "/" << totalConfigs << std::endl;
            std::cout << std::string(60, '=') << std::endl;
            
            // Adjust meta-GA parameters based on problem difficulty
            int metaPop = 20;
            int metaGen = 15;
            int trialsPerConfig = 5;
            int baseGen = 1000;
            
            // For 30D problems, use more careful evaluation
            if (dim == 30) {
                trialsPerConfig = 3; // Fewer trials but longer base GA runs
                baseGen = 2000;
            }
            
            // For Schwefel 30D, extra generations
            if (funcName == "Schwefel" && dim == 30) {
                baseGen = 3000;
            }
            
            // Create Meta-GA
            MetaGA metaGA(funcName, dim, funcMap[funcName], 
                         metaPop, metaGen, trialsPerConfig, baseGen);
            
            // Run optimization
            GAConfig bestConfig = metaGA.optimize();
            
            // Save to file
            std::string filename = "Results/optimized_params/OptimizedParams_" 
                                 + funcName + "_" + std::to_string(dim) + "D.txt";
            metaGA.saveConfig(bestConfig, filename);
            
            std::cout << "\nSaved optimized parameters to: " << filename << std::endl;
        }
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Meta-GA Optimization Complete!" << std::endl;
    std::cout << "All optimized parameters saved to Results/optimized_params/" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}
