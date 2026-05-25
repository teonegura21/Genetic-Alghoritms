#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <map>
#include "GeneticAlgorithm.h"
#include "ObjectiveFunctions.h"

// Structure to hold comparison results
struct ComparisonResult {
    std::string function;
    int dimension;
    
    // H2 Baseline (from original results)
    double h2_best;
    double h2_worst;
    double h2_mean;
    double h2_stddev;
    
    // Optimized Meta-GA
    double opt_best;
    double opt_worst;
    double opt_mean;
    double opt_stddev;
    
    // Improvement metrics
    double improvement_percent;
};

// Load optimized configuration from file
struct GAConfig {
    int populationSize;
    double mutationRate;
    double crossoverRate;
    double eta_c;
    double eta_m;
};

GAConfig loadConfig(const std::string& filename) {
    GAConfig config;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        
        if (key == "Population") {
            std::string size_str;
            iss >> size_str >> config.populationSize;
        } else if (key == "Mutation") {
            std::string rate_str;
            iss >> rate_str >> config.mutationRate;
        } else if (key == "Crossover") {
            std::string rate_str;
            iss >> rate_str >> config.crossoverRate;
        } else if (key == "Eta_c:") {
            iss >> config.eta_c;
        } else if (key == "Eta_m:") {
            iss >> config.eta_m;
        }
    }
    
    file.close();
    return config;
}

// Calculate statistics
struct Stats {
    double best, worst, mean, stddev;
};

Stats calculateStats(const std::vector<double>& results) {
    Stats stats;
    stats.best = results[0];
    stats.worst = results[0];
    double sum = 0.0;
    
    for (double val : results) {
        if (val < stats.best) stats.best = val;
        if (val > stats.worst) stats.worst = val;
        sum += val;
    }
    
    stats.mean = sum / results.size();
    
    double sqSum = 0.0;
    for (double val : results) {
        sqSum += (val - stats.mean) * (val - stats.mean);
    }
    stats.stddev = std::sqrt(sqSum / results.size());
    
    return stats;
}

int main() {
    std::cout << "=== Comparison: Meta-GA Optimized vs H2 Baseline ===" << std::endl;
    std::cout << "Running 30 trials for each configuration..." << std::endl << std::endl;
    
    std::vector<std::string> functions = {"DeJong1", "Schwefel", "Rastrigin", "Michalewicz"};
    std::vector<int> dimensions = {5, 10, 30};
    
    std::map<std::string, std::function<double(const std::vector<double>&)>> funcMap = {
        {"DeJong1", MathFunctions::DeJong1},
        {"Schwefel", MathFunctions::Schwefel},
        {"Rastrigin", MathFunctions::Rastrigin},
        {"Michalewicz", MathFunctions::Michalewicz}
    };
    
    // H2 Baseline results (from Rezultate H2.txt)
    std::map<std::string, double> h2Results = {
        {"DeJong1_5D", 0.000000}, {"DeJong1_10D", 0.000000}, {"DeJong1_30D", 0.000000},
        {"Schwefel_5D", -1925.152823}, {"Schwefel_10D", -3759.502918}, {"Schwefel_30D", -11089.007358},
        {"Rastrigin_5D", 0.000000}, {"Rastrigin_10D", 0.000002}, {"Rastrigin_30D", 0.000034},
        {"Michalewicz_5D", -4.686266}, {"Michalewicz_10D", -9.649157}, {"Michalewicz_30D", -29.355069}
    };
    
    std::vector<ComparisonResult> allResults;
    
    for (const auto& funcName : functions) {
        for (int dim : dimensions) {
            std::cout << "\n--- Testing: " << funcName << " " << dim << "D ---" << std::endl;
            
            // Load optimized config
            std::string configFile = "Results/optimized_params/OptimizedParams_" 
                                   + funcName + "_" + std::to_string(dim) + "D.txt";
            GAConfig config = loadConfig(configFile);
            
            std::cout << "Loaded Config - Pop: " << config.populationSize 
                     << ", MutRate: " << config.mutationRate 
                     << ", CrossRate: " << config.crossoverRate << std::endl;
            
            // Run 30 trials with optimized config
            int numRuns = 30;
            int maxGen = (dim == 30) ? 3000 : 1500;
            if (funcName == "Schwefel" && dim == 30) maxGen = 5000;
            
            double lowerBound, upperBound;
            MathFunctions::getBounds(funcName, lowerBound, upperBound);
            
            std::vector<double> runResults;
            runResults.reserve(numRuns);
            
            for (int i = 0; i < numRuns; ++i) {
                GeneticAlgorithm ga(
                    config.populationSize,
                    dim,
                    maxGen,
                    config.mutationRate,
                    config.crossoverRate,
                    lowerBound,
                    upperBound,
                    funcMap[funcName],
                    config.eta_c,
                    config.eta_m
                );
                
                Individual best = ga.run();
                runResults.push_back(best.fitness);
                
                if ((i + 1) % 5 == 0) std::cout << "." << std::flush;
            }
            std::cout << std::endl;
            
            Stats optStats = calculateStats(runResults);
            
            // Get H2 baseline
            std::string key = funcName + "_" + std::to_string(dim) + "D";
            double h2Mean = h2Results[key];
            
            // Calculate improvement
            double improvement = 0.0;
            if (h2Mean != 0.0) {
                improvement = ((h2Mean - optStats.mean) / std::abs(h2Mean)) * 100.0;
            }
            
            std::cout << "H2 Baseline Mean: " << h2Mean << std::endl;
            std::cout << "Optimized Mean: " << optStats.mean << std::endl;
            std::cout << "Improvement: " << improvement << "%" << std::endl;
            
            ComparisonResult result;
            result.function = funcName;
            result.dimension = dim;
            result.h2_mean = h2Mean;
            result.opt_best = optStats.best;
            result.opt_worst = optStats.worst;
            result.opt_mean = optStats.mean;
            result.opt_stddev = optStats.stddev;
            result.improvement_percent = improvement;
            
            allResults.push_back(result);
        }
    }
    
    // Save comparison report
    std::ofstream reportFile("Results/MetaGA_Comparison_Report.txt");
    reportFile << "=== Meta-GA vs H2 Baseline Comparison Report ===" << std::endl;
    reportFile << std::fixed << std::setprecision(6);
    
    for (const auto& res : allResults) {
        reportFile << "\n" << res.function << " " << res.dimension << "D:" << std::endl;
        reportFile << "  H2 Baseline Mean: " << res.h2_mean << std::endl;
        reportFile << "  Optimized Best: " << res.opt_best << std::endl;
        reportFile << "  Optimized Mean: " << res.opt_mean << std::endl;
        reportFile << "  Optimized Std Dev: " << res.opt_stddev << std::endl;
        reportFile << "  Improvement: " << res.improvement_percent << "%" << std::endl;
    }
    
    reportFile.close();
    std::cout << "\n\nComparison report saved to: Results/MetaGA_Comparison_Report.txt" << std::endl;
    
    return 0;
}
